'''
Select sensor and detect transmitter
'''
import random
import math
import copy
import time
import numpy as np
import pandas as pd
from numba import cuda
from scipy.spatial import distance
from scipy.stats import multivariate_normal, norm, entropy
from joblib import Parallel, delayed
from sensor import Sensor
from transmitter import Transmitter
from utility import read_config, ordered_insert#, print_results
from it_tool import InformationTheoryTool
#from cuda_kernals import o_t_approx_kernal, o_t_kernal, o_t_approx_dist_kernal
import plots


class SelectSensor:
    '''Near-optimal low-cost sensor selection

    Attributes:
        config (json):               configurations - settings and parameters
        sen_num (int):               the number of sensors
        grid_len (int):              the length of the grid
        grid_priori (np.ndarray):    the element is priori probability of hypothesis - transmitter
        grid_posterior (np.ndarray): the element is posterior probability of hypothesis - transmitter
        transmitters (list):         a list of Transmitter
        sensors (dict):              a dictionary of Sensor. less than 10% the # of transmitter
        data (ndarray):              a 2D array of observation data
        covariance (np.ndarray):     a 2D array of covariance. each data share a same covariance matrix
        mean_stds (dict):            assume sigal between a transmitter-sensor pair is normal distributed
        subset (dict):               a subset of all sensors
        subset_index (list):         the linear index of sensor in self.sensors
        meanvec_array (np.ndarray):  contains the mean vector of every transmitter, for CUDA
        TPB (int):                   thread per block
    '''
    def __init__(self, filename):
        self.config = read_config(filename)
        self.sen_num = int(self.config["sensor_number"])
        self.grid_len = int(self.config["grid_length"])
        self.grid_priori = np.zeros(0)
        self.grid_posterior = np.zeros(0)
        self.transmitters = []
        self.sensors = []
        self.data = np.zeros(0)
        self.covariance = np.zeros(0)
        self.init_transmitters()
        self.set_priori()
        self.means_stds = {}
        self.subset = {}
        self.subset_index = []
        self.meanvec_array = np.zeros(0)
        self.TPB = 32

    #@profile
    def init_from_real_data(self, cov_file, sensor_file, hypothesis_file):
        '''Init everything from collected real data
           1. init covariance matrix
           2. init sensors
           3. init mean and std between every pair of transmitters and sensors
        '''
        cov = pd.read_csv(cov_file, header=None, delimiter=' ')
        del cov[len(cov)]
        self.covariance = cov.values

        self.sensors = []
        with open(sensor_file, 'r') as f:
            max_gain = 0.5*len(self.transmitters)
            index = 0
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y, std, cost = int(line[0]), int(line[1]), float(line[2]), float(line[3])
                self.sensors.append(Sensor(x, y, std, cost, gain_up_bound=max_gain, index=index))
                index += 1

        with open(hypothesis_file, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                tran_x, tran_y = int(line[0]), int(line[1])
                sen_x, sen_y = int(line[2]), int(line[3])
                mean, std = float(line[4]), float(line[5])
                self.means_stds[(tran_x, tran_y, sen_x, sen_y)] = (mean, std)

        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            mean_vec = []
            for sensor in self.sensors:
                sen_x, sen_y = sensor.x, sensor.y
                mean_std = self.means_stds.get((tran_x, tran_y, sen_x, sen_y))
                mean_vec.append(mean_std[0])
            transmitter.mean_vec = np.array(mean_vec)
            #setattr(transmitter, 'multivariant_gaussian', multivariate_normal(mean=transmitter.mean_vec, cov=self.covariance))
        self.transmitters_to_array()
        #del self.means_stds  # in 64*64 grid offline case, need to delete means_stds and comment multivariant_gaussian to save memory. otherwise exceed 4GB limit of joblib
        print('init done!')


    def set_priori(self):
        '''Set priori distribution - uniform distribution
        '''
        uniform = 1./(self.grid_len * self.grid_len)
        self.grid_priori = np.full((self.grid_len, self.grid_len), uniform)
        self.grid_posterior = np.full((self.grid_len, self.grid_len), uniform)


    def init_transmitters(self):
        '''Initiate a transmitter at all locations
        '''
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                transmitter = Transmitter(i, j)
                setattr(transmitter, 'hypothesis', i*self.grid_len + j)
                self.transmitters.append(transmitter)


    def init_random_sensors(self):
        '''Initiate some sensors randomly
        '''
        noise_l, noise_h = float(self.config["noise_low"]), float(self.config["noise_high"])
        i = 0
        while i < self.sen_num:
            x = random.randint(0, self.grid_len-1) # randomly find a place for a sensor
            y = random.randint(0, self.grid_len-1)
            if self.exist_sensor(x, y):  # a sensor exists at (x, y)
                continue
            else:                        # no sensor exists at (x,y)
                self.sensors.append(Sensor(x, y, random.uniform(noise_l, noise_h)))  # the noise is here
                i += 1


    def exist_sensor(self, x, y):
        '''Test whether a sensor exists at (x, y)
        '''
        for sensor in self.sensors:
            if sensor.x == x and sensor.y == y:
                return True
        return False


    def save_sensor(self, filename):
        '''Save location of sensors
        '''
        with open(filename, 'w') as f:
            for sensor in self.sensors:
                f.write(sensor.output())


    def read_init_sensor(self, filename):
        '''Read location of sensors and init the sensors
        Attributes:
            filename (str)
        '''
        self.sensors = []
        max_gain = 0.5*len(self.transmitters)
        with open(filename, 'r') as f:
            index = 0
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y, std = int(line[0]), int(line[1]), float(line[2])
                self.sensors.append(Sensor(x, y, std, gain_up_bound=max_gain, index=index))
                index += 1


    def save_mean_std(self, filename):
        '''Save the mean and std of each transmitter-sensor pair.
           Mean is computed by f(x) = 100 - 30*math.log(2*dist)
        Attributes:
            filename (str)
        '''
        with open(filename, 'w') as f:
            for transmitter in self.transmitters:
                tran_x, tran_y = transmitter.x, transmitter.y
                for sensor in self.sensors:
                    sen_x, sen_y, std = sensor.x, sensor.y, sensor.std
                    dist = distance.euclidean([sen_x, sen_y], [tran_x, tran_y])
                    dist = 0.5 if dist < 1e-2 else dist  # in case distance is zero
                    mean = 100 - 33*math.log(2*dist)
                    mean = 0 if mean < 0 else mean
                    f.write("%d %d %d %d %f %f\n" % (tran_x, tran_y, sen_x, sen_y, mean, std))


    def read_mean_std(self, filename):
        '''read mean std information between transmitters and sensors
        Attributes:
            filename (str)
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                tran_x, tran_y = int(line[0]), int(line[1])
                sen_x, sen_y = int(line[2]), int(line[3])
                mean, std = float(line[4]), float(line[5])
                self.means_stds[(tran_x, tran_y, sen_x, sen_y)] = (mean, std)


    def generate_data(self, sample_file):
        '''Since we don't have the real data yet, we make up some artificial data according to mean_std.txt
           Then save them in a csv file. also save the mean vector
        Attributes:
            sample_file (str): filename for artificial sample
            mean_vec_file (str): filename for mean vector, the mean vector computed from sampled data
        '''
        transmitter = self.transmitters[0]
        tran_x, tran_y = transmitter.x, transmitter.y
        data = []
        i = 0
        while i < 5000:                  # sample 5000 times for a single transmitter
            one_transmitter = []
            for sensor in self.sensors:  # for each transmitter, send signal to all sensors
                sen_x, sen_y = sensor.x, sensor.y
                mean, std = self.means_stds.get((tran_x, tran_y, sen_x, sen_y))
                one_transmitter.append(np.random.normal(mean, std))
            data.append(one_transmitter)
            i += 1
        data_pd = pd.DataFrame(data)
        data_pd.to_csv(sample_file, index=False, header=False)


    def compute_multivariant_gaussian(self, sample_file):
        '''Read data and mean vectors, then compute the guassian function by using the data
           Each hypothesis corresponds to a single gaussian function
           with different mean but the same covariance.
        Attributes:
            sample_file (str)
            mean_vec_file (str)
        '''
        data = pd.read_csv(sample_file, header=None)
        self.covariance = np.cov(data.as_matrix().T)  # compute covariance matrix by date from one transmitter
        #print('Computed covariance!')                 # assume all transmitters share the same covariance

        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            mean_vec = []
            for sensor in self.sensors:
                sen_x, sen_y = sensor.x, sensor.y
                mean_std = self.means_stds.get((tran_x, tran_y, sen_x, sen_y))
                mean_vec.append(mean_std[0])
            transmitter.mean_vec = np.array(mean_vec)
            setattr(transmitter, 'multivariant_gaussian', multivariate_normal(mean=transmitter.mean_vec, cov=self.covariance))
        self.transmitters_to_array()

    def no_selection(self):
        '''The subset is all the sensors
        '''
        self.subset = copy.deepcopy(self.sensors)


    def update_subset(self, subset_index):
        '''Given a list of sensor indexes, which represents a subset of sensors, update self.subset
        Attributes:
            subset_index (list): a list of sensor indexes. guarantee sorted
        '''
        self.subset = []
        self.subset_index = subset_index
        for index in self.subset_index:
            self.subset.append(self.sensors[index])


    def update_transmitters(self):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub and multivariate gaussian function
        '''
        for transmitter in self.transmitters:
            transmitter.set_mean_vec_sub(self.subset_index)
            new_cov = self.covariance[np.ix_(self.subset_index, self.subset_index)]
            transmitter.multivariant_gaussian = multivariate_normal(mean=transmitter.mean_vec_sub, cov=new_cov)


    def update_mean_vec_sub(self, subset_index):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub
        Attributes:
            subset_index (list)
        '''
        for transmitter in self.transmitters:
            transmitter.set_mean_vec_sub(subset_index)


    def select_offline_random(self, number, cores):
        '''Select a subset of sensors randomly
        Attributes:
            number (int): number of sensors to be randomly selected
            cores (int): number of cores for parallelization
        Return:
            (list): results to be plotted. each element is (str, int, float),
                    where str is the list of selected sensors, int is # of sensor, float is O_T
        '''
        random.seed(0)
        subset_index = []
        plot_data = []
        sequence = [i for i in range(self.sen_num)]
        i = 1

        subset_to_compute = []
        while i <= number:
            select = random.choice(sequence)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            sequence.remove(select)
            i += 1

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_random)(subset_index) for subset_index in subset_to_compute)

        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data


    def inner_random(self, subset_index):
        '''Inner loop for random
        '''
        #o_t = self.o_t(subset_index)
        o_t = self.o_t_host(subset_index)
        return (subset_index, o_t)


    def covariance_sub(self, subset_index):
        '''Given a list of index of sensors, return the sub covariance matrix
        Attributes:
            subset_index (list): list of index of sensors. should be sorted.
        Return:
            (np.ndarray): a 2D sub covariance matrix
        '''
        sub_cov = self.covariance[np.ix_(subset_index, subset_index)]
        return sub_cov


    def o_t_p(self, subset_index, cores):
        '''(Parallelized version of o_t function) Given a subset of sensors T, compute the O_T
        Attributes:
            subset_index (list): a subset of sensors T, guarantee sorted
            cores (int): number of cores to do the parallel
        Return O_T
        '''
        if not subset_index:  # empty sequence are false
            return 0
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = None
        try:
            sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        except Exception as e:
            print(e)

        prob = Parallel(n_jobs=cores)(delayed(self.inner_o_t)(subset_index, sub_cov_inv, transmitter_i) for transmitter_i in self.transmitters)
        o_t = 0
        for i in prob:
            o_t += i
        return o_t


    def inner_o_t(self, subset_index, sub_cov_inv, transmitter_i):
        '''The inner loop for o_t function (for parallelization)
        '''
        i_x, i_y = transmitter_i.x, transmitter_i.y
        transmitter_i.set_mean_vec_sub(subset_index)
        prob_i = []
        for transmitter_j in self.transmitters:
            j_x, j_y = transmitter_j.x, transmitter_j.y
            if i_x == j_x and i_y == j_y:
                continue
            transmitter_j.set_mean_vec_sub(subset_index)
            pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
            prob_i.append(1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
        product = 1
        for i in prob_i:
            product *= i
        return product*self.grid_priori[i_x][i_y]

    #@profile
    def o_t(self, subset_index):
        '''Given a subset of sensors T, compute the O_T
        Attributes:
            subset_index (list): a subset of sensors T, guarantee sorted
        Return O_T
        '''
        if not subset_index:  # empty sequence are false
            return 0
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        o_t = 0
        for transmitter_i in self.transmitters:
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = 1
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i *= (1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            o_t += prob_i * self.grid_priori[i_x][i_y]
        return o_t


    def o_t_approximate(self, subset_index):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Attributes:
            subset_index (list): a subset of sensors T, needs guarantee sorted
        '''
        if not subset_index:  # empty sequence are false
            return -99999999999.
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        prob_error = 0                              # around 3% speed up by replacing [] to float

        for transmitter_i in self.transmitters:
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = 0
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i += norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)))
            prob_error += prob_i * self.grid_priori[i_x][i_y]
        return 1 - prob_error


    def o_t_approximate_2(self, subset_index):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Attributes:
            subset_index (list): a subset of sensors T, needs guarantee sorted
        '''
        if not subset_index:  # empty sequence are false
            return -99999999999.
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        prob_error = 0                              # around 3% speed up by replacing [] to float
        i = 0
        for transmitter_i in self.transmitters:
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = 0
            j = 0
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                tmp = norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)))
                prob_i += tmp
                print((i, j, tmp*self.grid_priori[i_x][i_y]))
                j += 1
            prob_error += prob_i * self.grid_priori[i_x][i_y]
            i += 1
        return 1 - prob_error


    def select_offline_greedy_p(self, budget, cores):
        '''(Parallel version) Select a subset of sensors greedily. offline + homo version
        Attributes:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        subset_to_compute = []
        while cost < budget and complement_index:
            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy)(subset_index, candidate) for candidate in complement_index)

            best_candidate = candidate_results[0][0]   # an element of candidate_results is a tuple - (int, float, list)
            maximum = candidate_results[0][1]          # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                print(candidate[2], candidate[1])
                if candidate[1] > maximum:
                    best_candidate = candidate[0]
                    maximum = candidate[1]

            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += 1
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), maximum, 0]) # don't compute real o_t now, delay to after all the subsets are selected

            if maximum > 0.999:
                break

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy_real_ot)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]
        return plot_data


    def select_offline_greedy_p_lazy(self, budget, cores, cuda_kernal=None):
        '''(Parallel + Lazy greedy) Select a subset of sensors greedily. offline + homo version
        Attributes:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        counter = 0
        base_ot_approx = 0
        if cuda_kernal == o_t_approx_kernal:
            base_ot_approx = 1 - 0.5*len(self.transmitters)
        elif cuda_kernal == o_t_approx_dist_kernal:
            largest_dist = (self.grid_len-1)*math.sqrt(2)
            max_gain_up_bound = 0.5*len(self.transmitters)*largest_dist   # the default bound is for non-distance
            for sensor in self.sensors:                                   # need to update the max gain upper bound for o_t_approx with distance
                sensor.gain_up_bound = max_gain_up_bound
            base_ot_approx = (1 - 0.5*len(self.transmitters))*largest_dist

        plot_data = []
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)    # S\T in the paper
        subset_to_compute = []
        while cost < budget and complement_sensors:
            best_candidate = -1
            best_sensor = None
            complement_sensors.sort()   # sorting the gain descendingly
            new_base_ot_approx = 0
            #for sensor in complement_sensors:
            #    print((sensor.index, sensor.gain_up_bound), end=' ')
            #print()
            update, max_gain = 0, 0
            while update < len(complement_sensors):
                update_end = update+cores if update+cores <= len(complement_sensors) else len(complement_sensors)
                candidiate_index = []
                for i in range(update, update_end):
                    candidiate_index.append(complement_sensors[i].index)
                counter += 1
                candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy)(subset_index, cuda_kernal, candidate) for candidate in candidiate_index)
                # an element of candidate_results is a tuple - (index, o_t_approx, subsetlist)
                for i, j in zip(range(update, update_end), range(0, cores)):  # the two range might be different, if the case, follow the first range
                    complement_sensors[i].gain_up_bound = candidate_results[j][1] - base_ot_approx  # update the upper bound of gain
                    #print(candidate_results[j][2], candidate_results[j][1], base_ot_approx, complement_sensors[i].gain_up_bound)
                    if complement_sensors[i].gain_up_bound > max_gain:
                        max_gain = complement_sensors[i].gain_up_bound
                        best_candidate = candidate_results[j][0]
                        best_sensor = complement_sensors[i]
                        new_base_ot_approx = candidate_results[j][1]

                if update_end < len(complement_sensors) and max_gain > complement_sensors[update_end].gain_up_bound:   # where the lazy happens
                    print('\n***LAZY!***\n', cost, (update, update_end), len(complement_sensors), '\n')
                    break
                update += cores
            base_ot_approx = new_base_ot_approx             # update the base o_t_approx for the next iteration
            print(best_candidate, subset_index, base_ot_approx, '\n')
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx, 0]) # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            cost += 1
        print('number of o_t_approx', counter)
        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy_real_ot)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]

        return plot_data


    def inner_greedy(self, subset_index, cuda_kernal, candidate):
        '''Inner loop for selecting candidates
        Attributes:
            subset_index (list):
            candidate (int):
        Return:
            (tuple): (index, o_t_approx, new subset_index)
        '''
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)     # guarantee subset_index always be sorted here
        #o_t = self.o_t_approximate(subset_index2)
        o_t = self.o_t_approx_host_2(subset_index2, cuda_kernal)
        return (candidate, o_t, subset_index2)


    def select_offline_greedy(self, budget):
        '''Select a subset of sensors greedily. offline + homo version
        Attributes:
            budget (int): budget constraint
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        plot_data = []

        while cost < budget and complement_index:
            maximum = self.o_t_approximate(subset_index)                # L in the paper
            best_candidate = complement_index[0]            # init the best candidate as the first one
            for candidate in complement_index:
                ordered_insert(subset_index, candidate)     # guarantee subset_index always be sorted here
                temp = self.o_t_approximate(subset_index)
                print(subset_index, temp)
                if temp > maximum:
                    maximum = temp
                    best_candidate = candidate
                subset_index.remove(candidate)
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            cost += 1

        return plot_data


    def select_offline_random_hetero(self, budget, cores):
        '''Offline selection when the sensors are heterogeneous

        Attributes:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
        '''
        '''
        energy = pd.read_csv('data/energy.txt', header=None)  # load the energy cost
        size = energy[1].count()
        i = 0
        for sensor in self.sensors:
            setattr(self.sensors.get(sensor), 'cost', energy[1][i%size])
            i += 1
        '''
        random.seed(0)    # though algorithm is random, the results are the same every time

        self.subset = {}
        subset_index = []
        plot_data = []
        sequence = [i for i in range(self.sen_num)]
        cost = 0
        cost_list = []
        subset_to_compute = []
        while cost < budget:
            option = []
            for index in sequence:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:                      # if there are no sensors that can be selected, then break
                break
            select = random.choice(option)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            sequence.remove(select)
            cost += self.sensors[select].cost
            cost_list.append(cost)

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_random)(subset_index) for subset_index in subset_to_compute)

        for cost, result in zip(cost_list, subset_results):
            plot_data.append((str(result[0]), cost, result[1]))

        return plot_data


    def select_offline_greedy_hetero(self, budget, cores):
        '''Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two

        Attributes:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
        #energy = pd.read_csv('data/energy.txt', header=None)  # load the energy cost
        #size = energy[1].count()
        #i = 0
        #for sensor in self.sensors:
        #    setattr(sensor, 'cost', energy[1][i%size])
        #    i += 1
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        maximum = 0
        first_pass_plot_data = []
        while cost < budget and complement_index:
            option = []
            for index in complement_index:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:                      # if there are no sensors that can be selected, then break
                break

            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy)(subset_index, candidate) for candidate in option)

            best_candidate = candidate_results[0][0]   # an element of candidate_results is a tuple - (int, float, list)
            maximum = candidate_results[0][1]          # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                #print(candidate[2], candidate[1])
                if candidate[1] > maximum:
                    best_candidate = candidate[0]
                    maximum = candidate[1]

            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += self.sensors[best_candidate].cost
            first_pass_plot_data.append([copy.deepcopy(subset_index), cost, 0])           # Y value is real o_t
            print(subset_index, maximum, cost)
            if maximum > 0.999:
                break

        print('end of the first homo pass and start of the second hetero pass')

        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        base_ot = 1 - 0.5*len(self.transmitters)            # O_T from the previous iteration
        second_pass_plot_data = []
        while cost < budget and complement_index:
            option = []
            for index in complement_index:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:
                break

            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy)(subset_index, candidate) for candidate in option)

            best_candidate = candidate_results[0][0]                       # an element of candidate_results is a tuple - (int, float, list)
            cost_of_candiate = self.sensors[best_candidate].cost
            new_base_ot = candidate_results[0][1]
            maximum = (candidate_results[0][1]-base_ot)/cost_of_candiate   # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                incre = candidate[1] - base_ot
                cost_of_candiate = self.sensors[candidate[0]].cost
                incre_cost = incre/cost_of_candiate     # increment of O_T devided by cost
                #print(candidate[2], candidate[1], incre, cost_of_candiate, incre_cost)
                if incre_cost > maximum:
                    best_candidate = candidate[0]
                    maximum = incre_cost
                    new_base_ot = candidate[1]
            base_ot = new_base_ot
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += self.sensors[best_candidate].cost
            second_pass_plot_data.append([copy.deepcopy(subset_index), cost, 0])           # Y value is real o_t
            print(subset_index, base_ot, cost)

        first_pass = []
        for data in first_pass_plot_data:
            first_pass.append(data[0])
        second_pass = []
        for data in second_pass_plot_data:
            second_pass.append(data[0])

        first_pass_o_ts = Parallel(n_jobs=cores)(delayed(self.inner_greedy_real_ot)(subset_index) for subset_index in first_pass)
        second_pass_o_ts = Parallel(n_jobs=cores)(delayed(self.inner_greedy_real_ot)(subset_index) for subset_index in second_pass)

        for i in range(len(first_pass_o_ts)):
            first_pass_plot_data[i][2] = first_pass_o_ts[i]
        for i in range(len(second_pass_o_ts)):
            second_pass_plot_data[i][2] = second_pass_o_ts[i]

        first_final_o_t = first_pass_plot_data[len(first_pass_plot_data)-1][2]
        second_final_o_t = second_pass_plot_data[len(second_pass_plot_data)-1][2]

        if second_final_o_t > first_final_o_t:
            print('second pass is selected')
            return second_pass_plot_data
        else:
            print('first pass is selected')
            return first_pass_plot_data


    def select_offline_greedy_hetero_lazy(self, budget, cores):
        '''(Lazy) Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two

        Attributes:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
        #energy = pd.read_csv('data/energy.txt', header=None)  # load the energy cost
        #size = energy[1].count()
        #i = 0
        #lowest_cost = 1
        #for sensor in self.sensors:
        #    setattr(sensor, 'cost', energy[1][i%size])
        #    if sensor.cost < lowest_cost:
        #        lowest_cost = sensor.cost
        #    i += 1

        base_ot_approx = 1 - 0.5*len(self.transmitters)
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)    # S\T in the paper
        first_pass_plot_data = []
        while cost < budget and complement_sensors:
            complement_sensors.sort()           # sort the sensors by gain upper bound descendingly
            option = []
            for sensor in complement_sensors:
                temp_cost = sensor.cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(sensor)
            if not option:                      # if there are no sensors that can be selected, then break
                break
            best_candidate = -1
            best_sensor = None
            new_base_ot_approx = 0
            update, max_gain = 0, 0
            while update < len(option):
                update_end = update+cores if update+cores <= len(option) else len(option)
                candidiate_index = []
                for i in range(update, update_end):
                    candidiate_index.append(option[i].index)

                candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy)(subset_index, candidate) for candidate in candidiate_index)
                # an element of candidate_results is a tuple - (index, o_t_approx, subsetlist)
                for i, j in zip(range(update, update_end), range(0, cores)):  # the two range might be different, if the case, follow the first range
                    complement_sensors[i].gain_up_bound = candidate_results[j][1] - base_ot_approx  # update the upper bound of gain
                    if complement_sensors[i].gain_up_bound > max_gain:
                        max_gain = complement_sensors[i].gain_up_bound
                        best_candidate = candidate_results[j][0]
                        best_sensor = complement_sensors[i]
                        new_base_ot_approx = candidate_results[j][1]

                if update_end < len(complement_sensors) and max_gain > complement_sensors[update_end].gain_up_bound:   # where the lazy happens
                    print('\n******LAZY!')
                    print(cost, (update, update_end), len(complement_sensors), '\n******\n')
                    break
                update += cores

            base_ot_approx = new_base_ot_approx
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_sensors.remove(best_sensor)
            cost += self.sensors[best_candidate].cost
            first_pass_plot_data.append([copy.deepcopy(subset_index), cost, 0])           # Y value is real o_t
            print(subset_index, base_ot_approx, cost)

        print('end of the first homo pass and start of the second hetero pass')

        i = 0
        lowest_cost = 1
        for sensor in self.sensors:
            if sensor.cost < lowest_cost:
                lowest_cost = sensor.cost
            i += 1

        max_gain_up_bound = 0.5*len(self.transmitters)/lowest_cost
        for sensor in self.sensors:
            sensor.gain_up_bound = max_gain_up_bound
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_sensors = copy.deepcopy(self.sensors)    # S\T in the paper
        base_ot_approx = 1 - 0.5*len(self.transmitters)
        second_pass_plot_data = []
        while cost < budget and complement_sensors:
            complement_sensors.sort()
            option = []
            for sensor in complement_sensors:
                temp_cost = sensor.cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(sensor)
            if not option:
                break
            best_candidate = -1
            best_sensor = None
            new_base_ot_approx = 0
            update, max_gain = 0, 0
            while update < len(option):
                update_end = update+cores if update+cores <= len(complement_sensors) else len(complement_sensors)
                candidiate_index = []
                for i in range(update, update_end):
                    candidiate_index.append(complement_sensors[i].index)

                candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy)(subset_index, candidate) for candidate in candidiate_index)
                # an element of candidate_results is a tuple - (index, o_t_approx, subsetlist)
                for i, j in zip(range(update, update_end), range(0, cores)):  # the two range might be different, if the case, follow the first range
                    complement_sensors[i].gain_up_bound = (candidate_results[j][1] - base_ot_approx)/complement_sensors[i].cost  # update the upper bound of gain
                    if complement_sensors[i].gain_up_bound > max_gain:
                        max_gain = complement_sensors[i].gain_up_bound
                        best_candidate = candidate_results[j][0]
                        best_sensor = complement_sensors[i]
                        new_base_ot_approx = candidate_results[j][1]

                if update_end < len(complement_sensors) and max_gain > complement_sensors[update_end].gain_up_bound:   # where the lazy happens
                    print('\n******LAZY!')
                    print(cost, (update, update_end), len(complement_sensors), '\n******\n')
                    break
                update += cores

            base_ot_approx = new_base_ot_approx             # update the base o_t_approx for the next iteration
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_sensors.remove(best_sensor)
            cost += self.sensors[best_candidate].cost
            second_pass_plot_data.append([copy.deepcopy(subset_index), cost, 0])           # Y value is real o_t
            print(subset_index, base_ot_approx, cost)

        first_pass = []
        for data in first_pass_plot_data:
            first_pass.append(data[0])
        second_pass = []
        for data in second_pass_plot_data:
            second_pass.append(data[0])

        first_pass_o_ts = Parallel(n_jobs=cores)(delayed(self.inner_greedy_real_ot)(subset_index) for subset_index in first_pass)
        second_pass_o_ts = Parallel(n_jobs=cores)(delayed(self.inner_greedy_real_ot)(subset_index) for subset_index in second_pass)

        for i in range(len(first_pass_o_ts)):
            first_pass_plot_data[i][2] = first_pass_o_ts[i]
        for i in range(len(second_pass_o_ts)):
            second_pass_plot_data[i][2] = second_pass_o_ts[i]

        first_final_o_t = first_pass_plot_data[len(first_pass_plot_data)-1][2]
        second_final_o_t = second_pass_plot_data[len(second_pass_plot_data)-1][2]

        if second_final_o_t > first_final_o_t:
            print('second pass is selected')
            return second_pass_plot_data
        else:
            print('first pass is selected')
            return first_pass_plot_data


    def inner_greedy_real_ot(self, subset_index):
        '''Compute the real o_t (accruacy of prediction)
        '''
        #o_t = self.o_t(subset_index)
        o_t = self.o_t_host(subset_index)
        return o_t


    def select_offline_coverage(self, budget, cores):
        '''A coverage-based baseline algorithm
        '''
        random.seed(0)
        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        first_sensor = None
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
                first_sensor = sensor
            i += 1
        subset_index = [first_index]
        subset_to_compute = [copy.deepcopy(subset_index)]
        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)

        radius = self.compute_coverage_radius(first_sensor, subset_index) # compute the radius
        print('radius', radius)
        coverage = np.zeros((self.grid_len, self.grid_len), dtype=int)
        self.add_coverage(coverage, first_sensor, radius)
        cost = 1
        while cost < budget and complement_index:  # find the sensor that has the least overlap
            least_overlap = 99999
            best_candidate = []
            best_sensor = []
            for candidate in complement_index:
                sensor = self.index_to_sensor(candidate)
                overlap = self.compute_overlap(coverage, sensor, radius)
                if overlap < least_overlap:
                    least_overlap = overlap
                    best_candidate = [candidate]
                    best_sensor = [sensor]
                elif overlap == least_overlap:
                    best_candidate.append(candidate)
                    best_sensor.append(sensor)
            choose = random.choice(range(len(best_candidate)))
            ordered_insert(subset_index, best_candidate[choose])
            complement_index.remove(best_candidate[choose])
            self.add_coverage(coverage, best_sensor[choose], radius)
            subset_to_compute.append(copy.deepcopy(subset_index))
            cost += 1

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_random)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for result in subset_results:
            plot_data.append((str(result[0]), len(result[0]), result[1]))

        return plot_data


    def select_offline_coverage_hetero(self, budget, cores):
        '''A coverage-based baseline algorithm (heterogeneous version)
        '''
        random.seed(0)

        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        first_sensor = None
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
                first_sensor = sensor
            i += 1
        subset_index = [first_index]
        subset_to_compute = [copy.deepcopy(subset_index)]
        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)

        radius = self.compute_coverage_radius(first_sensor, subset_index) # compute the radius
        print('radius', radius)

        coverage = np.zeros((self.grid_len, self.grid_len), dtype=int)
        self.add_coverage(coverage, first_sensor, radius)
        cost = self.sensors[first_index].cost
        cost_list = [cost]

        while cost < budget and complement_index:
            option = []
            for index in complement_index:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:                      # if there are no sensors that can be selected, then break
                break

            min_overlap_cost = 99999   # to minimize overlap*cost
            best_candidate = []
            best_sensor = []
            for candidate in option:
                sensor = self.index_to_sensor(candidate)
                overlap = self.compute_overlap(coverage, sensor, radius)
                temp_cost = self.sensors[candidate].cost
                overlap_cost = (overlap+0.001)*temp_cost
                if overlap_cost < min_overlap_cost:
                    min_overlap_cost = overlap_cost
                    best_candidate = [candidate]
                    best_sensor = [sensor]
                elif overlap_cost == min_overlap_cost:
                    best_candidate.append(candidate)
                    best_sensor.append(sensor)
            choose = random.choice(range(len(best_candidate)))
            ordered_insert(subset_index, best_candidate[choose])
            complement_index.remove(best_candidate[choose])
            self.add_coverage(coverage, best_sensor[choose], radius)
            subset_to_compute.append(copy.deepcopy(subset_index))
            cost += self.sensors[best_candidate[choose]].cost
            cost_list.append(cost)

        print(len(subset_to_compute), subset_to_compute)
        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_random)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for cost, result in zip(cost_list, subset_results):
            plot_data.append((str(result[0]), cost, result[1]))

        return plot_data


    def compute_coverage_radius(self, first_sensor, subset_index):
        '''Compute the coverage radius for the coverage-based selection algorithm
        Attibutes:
            first_sensor (tuple): sensor that is closest to the center
            subset_index (list):
        '''
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse
        radius = 1
        for i in range(1, int(self.grid_len/2)):    # compute 'radius'
            transmitter_i = self.transmitters[(first_sensor.x - i)*self.grid_len + first_sensor.y] # 2D index --> 1D index
            i_x, i_y = transmitter_i.x, transmitter_i.y
            if i_x < 0:
                break
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = []
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = transmitter_j.mean_vec_sub - transmitter_i.mean_vec_sub
                prob_i.append(1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            product = 1
            for prob in prob_i:
                product *= prob
            print(i, product)
            if product > 0.00001:     # set threshold
                radius = i
            else:
                break
        return radius


    def compute_overlap(self, coverage, sensor, radius):
        '''Compute the overlap between selected sensors and the new sensor
        Attributes:
            coverage (2D array)
            sensor (Sensor)
            radius (int)
        '''
        x_low = sensor.x - radius if sensor.x - radius >= 0 else 0
        x_high = sensor.x + radius if sensor.x + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor.y - radius if sensor.y - radius >= 0 else 0
        y_high = sensor.y + radius if sensor.y + radius <= self.grid_len-1 else self.grid_len-1

        overlap = 0
        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high):
                if distance.euclidean([x, y], [sensor.x, sensor.y]) <= radius:
                    overlap += coverage[x][y]
        return overlap


    def index_to_sensor(self, index):
        '''A temporary solution for the inappropriate data structure for self.sensors
        '''
        i = 0
        for sensor in self.sensors:
            if i == index:
                return sensor
            else:
                i += 1

    def add_coverage(self, coverage, sensor, radius):
        '''When seleted a sensor, add coverage by 1
        Attributes:
            coverage (2D array): each element is a counter for coverage
            sensor (Sensor): (x, y)
            radius (int): radius of a sensor
        '''
        x_low = sensor.x - radius if sensor.x - radius >= 0 else 0
        x_high = sensor.x + radius if sensor.x + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor.y - radius if sensor.y - radius >= 0 else 0
        y_high = sensor.y + radius if sensor.y + radius <= self.grid_len-1 else self.grid_len-1

        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high+1):
                if distance.euclidean([x, y], [sensor.x, sensor.y]) <= radius:
                    coverage[x][y] += 1


    def test_error(self):
        '''Generate new data, calculate posterior probability, compute classification error.
           For each transmitter, test 10 times
        '''
        total_test = 0
        error = 0
        self.grid_posterior = np.zeros((self.grid_len, self.grid_len))
        for transmitter in self.transmitters:   # test a transmitter
            transmitter.error = 0
            tran_x, tran_y = transmitter.x, transmitter.y
            if tran_x == tran_y:
                print(tran_x)
            i = 0
            while i < 10:  # test 10 times for each transmitter
                data = []
                for sensor in self.subset:
                    sen_x, sen_y = sensor.x, sensor.y
                    mean, std = self.means_stds.get((tran_x, tran_y, sen_x, sen_y))
                    data.append(np.random.normal(mean, std))
                for transmitter2 in self.transmitters:  # given hypothesis, the probability of data
                    multivariant_gaussian = transmitter2.multivariant_gaussian # see which hypothesis is "best"
                    tran_x2, tran_y2 = transmitter2.x, transmitter2.y
                    likelihood = multivariant_gaussian.pdf(data)
                    self.grid_posterior[tran_x2][tran_y2] = likelihood * self.grid_priori[tran_x2][tran_y2]
                denominator = self.grid_posterior.sum()   # we could neglect denominator
                if denominator <= 0:
                    continue
                self.grid_posterior = self.grid_posterior/denominator
                index_max = np.argmax(self.grid_posterior)
                max_x, max_y = self.index_inverse(index_max)
                if max_x != tran_x or max_y != tran_y:
                    error += 1
                    transmitter.add_error()
                total_test += 1
                i += 1

        return float(error)/total_test


    def index_inverse(self, index):
        '''Convert 1D index into 2D index
        '''
        x = int(index/self.grid_len)
        y = index%self.grid_len
        return (x, y)


    def select_online_greedy_hetero(self, budget, cores, true_index):
        '''Heterogeneous version of online greedy selection
        Attributes:
            budget (int): amount of budget, in the homo case, every sensor has budget=1
            cores (int): number of cores used in the parallezation
            cost_filename (str): file that has the cost of sensors
        '''
        self.set_priori()
        random.seed(1)
        plot_data = []
        true_transmitter = self.transmitters[true_index]         # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)

        number_hypotheses = 10*len(self.transmitters)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        subset_to_compute = []
        cost = 0
        cost_list = []
        while cost < budget and complement_index:
            print(cost, budget)
            option = []
            for index in complement_index:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:
                    option.append(index)
            if not option:
                break

            true_hypotheses = self.generate_true_hypotheses(number_hypotheses)
            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_online_greedy)(subset_index, true_hypotheses, candidate) \
                                for candidate in option)

            best_candidate = candidate_results[0][0]
            cost_of_candidate = self.sensors[best_candidate].cost
            maximum = candidate_results[0][1]/cost_of_candidate
            for candidate in candidate_results:
                mi_incre = candidate[1]
                cost_of_candidate = self.sensors[candidate[0]].cost
                mi_incre_cost = mi_incre/cost_of_candidate
                print(candidate[0], mi_incre, cost_of_candidate, mi_incre_cost)
                if mi_incre_cost > maximum:
                    maximum = mi_incre_cost
                    best_candidate = candidate[0]
            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += self.sensors[best_candidate].cost
            cost_list.append(cost)
            subset_to_compute.append(copy.deepcopy(subset_index))

        print(len(subset_to_compute), subset_to_compute)
        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for cost, result in zip(cost_list, subset_results):
            plot_data.append((str(result[0]), cost, result[1]))

        return plot_data


    def select_online_greedy_p(self, budget, cores, true_index):
        '''(Parallel version) of online greedy selection
        Attributes:
            budget (int): amount of budget, in the homo case, every sensor has budget=1
            cores (int): number of cores used in the parallezation
        '''
        self.set_priori()
        plot_data = []
        random.seed(1)
        true_transmitter = self.transmitters[true_index] # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        self.print_grid(self.grid_priori)
        number_hypotheses = 10*len(self.transmitters)
        cost = 0
        subset_to_compute = []
        while cost < budget and complement_index:
            true_hypotheses = self.generate_true_hypotheses(number_hypotheses)
            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_online_greedy)(subset_index, true_hypotheses, candidate) \
                                for candidate in complement_index)

            best_candidate = candidate_results[0][0]
            maximum = candidate_results[0][1]
            for candidate in candidate_results:
                print(candidate[2], candidate[1])
                if candidate[1] > maximum:
                    maximum = candidate[1]
                    best_candidate = candidate[0]

            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            subset_to_compute.append(copy.deepcopy(subset_index))
            print('MI = ', maximum)
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += 1

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)

        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data


    def select_online_greedy_p_2(self, budget, cores, true_index):
        '''(Parallel version) Version 2 of online greedy selection with mutual_information version 2
        Attributes:
            budget (int): amount of budget, in the homo case, every sensor has budget=1
            cores (int): number of cores used in the parallezation
            true_index (int): the true transmitter
        Return:
            plot_data (list)
        '''
        self.set_priori()
        plot_data = []
        random.seed(2)
        true_transmitter = self.transmitters[true_index] # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        self.print_grid(self.grid_priori)
        init_priori = [self.grid_priori[0][0]] * (self.grid_len*self.grid_len)
        h_h = entropy(init_priori, base=2)
        #number_hypotheses = 10*len(self.transmitters)
        cost = 0
        subset_to_compute = []
        while cost < budget and complement_index:
            #true_hypotheses = self.generate_true_hypotheses(number_hypotheses)
            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_online_greedy_2)(subset_index, true_transmitter, h_h, candidate) \
                                for candidate in complement_index)

            best_candidate = candidate_results[0][0]
            maximum = candidate_results[0][1]
            for candidate in candidate_results:
                print(candidate[2], candidate[1])
                if candidate[1] > maximum:
                    maximum = candidate[1]
                    best_candidate = candidate[0]

            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            subset_to_compute.append(copy.deepcopy(subset_index))
            print('MI = ', maximum)
            h_h -= maximum
            self.print_subset(subset_index)
            self.update_hypothesis_2(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += 1

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)

        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data



    def inner_online_greedy(self, subset_index, true_hypotheses, candidate):
        '''The inner loop for online greedy
        Attributes:
            subset_index (list):
            candidate (int):
            true_hypotheses (list): a list of int
        Return:
            (tuple): (index, mutual information, new subset_index) -- (int, float, list)
        '''
        np.random.seed(candidate)
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)
        mi = self.mutual_information(subset_index2, true_hypotheses)
        return (candidate, mi, subset_index2)


    def inner_online_greedy_2(self, subset_index, true_transmitter, h_h, candidate):
        '''The inner loop for online greedy version 2
        Attributes:
            subset_index (list)
            true_transmitter (int)
            h_h (float) the H(H) from equation #10
        '''
        np.random.seed(candidate)
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)
        mi = self.mutual_information_2(subset_index2, true_transmitter, h_h)
        return (candidate, mi, subset_index2)


    #@profile
    def select_online_greedy(self, budget, true_index):
        '''The online greedy selection. Homogeneous.
        Attributes:
            budget (int)
            true_index (int)
        '''
        plot_data = []
        random.seed(1)
        np.random.seed(2)
        #rand = random.randint(0, self.grid_len*self.grid_len-1)
        true_transmitter = self.transmitters[true_index]         # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        self.print_grid(self.grid_priori)
        number_hypotheses = 10*len(self.transmitters)
        cost = 0

        while cost < budget and complement_index:
            true_hypotheses = self.generate_true_hypotheses(number_hypotheses)
            maximum = self.mutual_information(subset_index, true_hypotheses)
            best_candidate = complement_index[0]
            for candidate in complement_index:
                ordered_insert(subset_index, candidate)
                mi = self.mutual_information(subset_index, true_hypotheses)
                print(subset_index, 'MI =', mi)
                if mi > maximum:
                    maximum = mi
                    best_candidate = candidate
                subset_index.remove(candidate)
            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            print('MI = ', maximum)
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += 1
        return plot_data

    #@profile
    def select_online_greedy_2(self, budget, true_index):
        '''The online greedy selection version 2. Homogeneous.
        Attributes:
            budget (int)
            true_index (int)
        '''
        plot_data = []
        random.seed(1)
        np.random.seed(2)
        #rand = random.randint(0, self.grid_len*self.grid_len-1)
        true_transmitter = self.transmitters[true_index]         # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        self.print_grid(self.grid_priori)
        init_priori = [self.grid_priori[0][0]] * (self.grid_len*self.grid_len)
        h_h = entropy(init_priori, base=2)
        cost = 0

        while cost < budget and complement_index:
            maximum = self.mutual_information_2(subset_index, true_transmitter, h_h)
            best_candidate = complement_index[0]
            for candidate in complement_index:
                ordered_insert(subset_index, candidate)
                mi = self.mutual_information_2(subset_index, true_transmitter, h_h)
                print(subset_index, 'MI =', mi)
                if mi > maximum:
                    maximum = mi
                    best_candidate = candidate
                subset_index.remove(candidate)
            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            print('MI = ', maximum)
            h_h -= maximum
            self.print_subset(subset_index)
            self.update_hypothesis_2(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += 1
        return plot_data


    def print_subset(self, subset_index):
        '''Print the subset_index and its 2D location
        Attriubtes:
            subset_index (list)
        '''
        print(subset_index, end=' ')
        print('[', end=' ')
        for index in subset_index:
            print((self.sensors[index].x, self.sensors[index].y), end=' ')
        print(']')


    def print_grid(self, grid):
        '''Print priori or posterior grid
        '''
        size = len(grid)
        print('')
        for i in range(size):
            print('[', end=' ')
            for j in range(size):
                print('%.5f' % grid[i][j], end=' ')
            print(']')
        print('')


    def update_hypothesis(self, true_transmitter, subset_index):
        '''Use Bayes formula to update P(hypothesis): from prior to posterior
           After we add a new sensor and get a larger subset, the larger subset begins to observe data from true transmitter
        Attributes:
            true_transmitter (Transmitter)
            subset_index (list)
        '''
        self.subset_index = subset_index
        self.update_transmitters()
        true_x, true_y = true_transmitter.x, true_transmitter.y
        np.random.seed(true_x*self.grid_len + true_y)
        data = []                          # the true transmitter generate some data
        for index in subset_index:
            sensor = self.sensors[index]
            mean, std = self.means_stds.get((true_x, true_y, sensor.x, sensor.y))
            data.append(np.random.normal(mean, std))
        for trans in self.transmitters:
            likelihood = trans.multivariant_gaussian.pdf(data)
            self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
        denominator = self.grid_posterior.sum()
        try:
            self.grid_posterior = self.grid_posterior/denominator
            self.grid_priori = copy.deepcopy(self.grid_posterior)   # the posterior in this iteration will be the prior in the next iteration
        except Exception as e:
            print(e)
            print('denominator', denominator)


    def update_hypothesis_2(self, true_transmitter, subset_index):
        '''Use Bayes formula to update P(hypothesis): from prior to posterior
           After we add a new sensor and get a larger subset, the larger subset begins to observe data from true transmitter
           An important update from update_hypothesis to update_hypothesis_2 is that we are not using attribute transmitter.multivariant_gaussian. It saves money
        Attributes:
            true_transmitter (Transmitter)
            subset_index (list)
        '''
        true_x, true_y = true_transmitter.x, true_transmitter.y
        np.random.seed(true_x*self.grid_len + true_y*true_y)  # change seed here
        data = []                          # the true transmitter generate some data
        for index in subset_index:
            sensor = self.sensors[index]
            mean, std = self.means_stds.get((true_x, true_y, sensor.x, sensor.y))
            data.append(np.random.normal(mean, std))
        for trans in self.transmitters:
            trans.set_mean_vec_sub(subset_index)
            cov_sub = self.covariance[np.ix_(subset_index, subset_index)]
            likelihood = multivariate_normal(mean=trans.mean_vec_sub, cov=cov_sub).pdf(data)
            self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
        denominator = self.grid_posterior.sum()
        try:
            self.grid_posterior = self.grid_posterior/denominator
            self.grid_priori = copy.deepcopy(self.grid_posterior)   # the posterior in this iteration will be the prior in the next iteration
        except Exception as e:
            print(e)
            print('denominator', denominator)


    def generate_true_hypotheses(self, number):
        '''Generate true hypotheses according to self.grid_priori
        Attributes:
            number (int): the number of true hypothesis we are generating
        Return:
            (list): a list of true hypotheis
        '''
        true_hypotheses = []
        grid = copy.deepcopy(self.grid_priori)
        grid *= number
        hypothesis = 0
        for x in range(self.grid_len):
            for y in range(self.grid_len):
                if grid[x][y] > 1e-3:     # if there is less than 0.001 transmitter at the place, then ignore it
                    repeat = math.ceil(grid[x][y])
                    i = 0
                    while i < repeat:
                        true_hypotheses.append(hypothesis)
                        i += 1
                hypothesis += 1
        random.shuffle(true_hypotheses)
        size = len(true_hypotheses)
        if size > number:                 # remove redundant hypotheses till desired amount of hypotheses
            i = 0
            while i < size - number:
                true_hypotheses.pop()
                i += 1
        elif size < number:
            i = 0
            while i < number - size:
                true_hypotheses.append(true_hypotheses[i])
                i += 1
        return true_hypotheses

    #@profile
    def mutual_information(self, subset_index, true_hypotheses):
        '''Mutual information between the observation of a subset of sensors and true hypothesis
        Attributes:
            subset_index (list): the X_{T,sk} in Algorithm-2,  R2 in the I(R1,R2) formula
            true_hypotheses (list): the Y in Algorithm-2,      R1 in the I(R1,R2) formula
        Return:
            (float) mutual information
        '''
        if not subset_index:
            return 0
        self.subset_index = subset_index
        self.update_transmitters()
        sensor_observe = []                  # R2 in the paper
        for hypothesis in true_hypotheses:   # R1 in the paper
            true_transmitter = self.transmitters[hypothesis]
            true_x, true_y = true_transmitter.x, true_transmitter.y
            data = []                        # data generated from true transmitter
            for index in subset_index:
                sensor = self.sensors[index]
                mean, std = self.means_stds.get((true_x, true_y, sensor.x, sensor.y))
                data.append(np.random.normal(mean, std))
            for trans in self.transmitters:
                likelihood = trans.multivariant_gaussian.pdf(data)
                self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
            #self.print_grid(self.grid_posterior)
            r2 = np.argmax(self.grid_posterior)
            sensor_observe.append(r2)
        data = np.array([true_hypotheses, sensor_observe])
        it_tool = InformationTheoryTool(data)
        return it_tool.mutual_information(0, 1)

    #@profile
    def mutual_information_2(self, subset_index, true_transmitter, h_h):
        '''the 2nd version of mutual information without generating data many many times. just generate data once in the Bayes formula
        Attributes:
            subset_index (list): the X_{T}
            true_transmitter (int): the true transmitter
            h_h (float): H(h), see equation #10 in the paper
        Return:
            (float) mutual information
        '''
        if not subset_index:
            return 0
        posterior = np.zeros(self.grid_len*self.grid_len)  # compute the entropy from the "temporary posterior grid"
        np.random.seed(true_transmitter.x*self.grid_len + true_transmitter.y)
        data = []   # the true transmitter generate some data
        for index in subset_index:
            sensor = self.sensors[index]
            mean, std = self.means_stds.get((true_transmitter.x, true_transmitter.y, sensor.x, sensor.y))
            data.append(np.random.normal(mean, std))

        cov_sub = self.covariance[np.ix_(subset_index, subset_index)]
        for trans in self.transmitters:
            trans.set_mean_vec_sub(subset_index)
            likelihood = multivariate_normal(mean=trans.mean_vec_sub, cov=cov_sub).pdf(data)
            posterior[trans.x * self.grid_len + trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
        posterior = posterior/posterior.sum()
        return h_h - entropy(posterior, base=2)

    #@profile
    def accuracy(self, subset_index, true_transmitter):
        '''Test the accuracy of a subset of sensors when detecting the (single) true transmitter
        Attributes:
            subset_index (list):
            true_transmitter (Transmitter):
        '''
        self.set_priori()
        self.subset_index = subset_index
        self.update_transmitters()
        true_x, true_y = true_transmitter.x, true_transmitter.y
        seed = true_x*self.grid_len + true_y
        np.random.seed(seed)
        random.seed(seed)
        test_num = 1000   # test a thousand times
        success = 0
        i = 0
        while i < test_num:
            data = []
            for index in subset_index:
                sensor = self.sensors[index]
                mean, std = self.means_stds.get((true_x, true_y, sensor.x, sensor.y))
                data.append(np.random.normal(mean, std))
            for trans in self.transmitters:
                likelihood = trans.multivariant_gaussian.pdf(data)
                self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
            #self.print_grid(self.grid_posterior)
            max_posterior = np.argwhere(self.grid_posterior == np.amax(self.grid_posterior))
            for max_post in max_posterior:  # there might be multiple places with the same highest posterior
                if max_post[0] == true_x and max_post[1] == true_y:
                    count = len(max_posterior)
                    if random.randint(1, count) == 1: # when true transmitter is among the max posterior, randomly pick one
                        success += 1
            i += 1
        return float(success)/test_num


    def select_online_random(self, budget, cores, true_index):
        '''The online random selection
        Attributes:
            budget (int):
            cores (int):
        '''
        self.set_priori()
        random.seed(5)
        np.random.seed(5)
        true_transmitter = self.transmitters[true_index]         # in online selection, there is true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        plot_data = []
        subset_to_compute = []
        cost = 0

        while cost < budget and complement_index:
            select = random.choice(complement_index)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            complement_index.remove(select)
            cost += 1

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)
        print(subset_results)
        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data


    def inner_online_accuracy(self, true_transmitter, subset_index):
        '''The inner loop for online random
        '''
        accuracy = self.accuracy(subset_index, true_transmitter)
        return (subset_index, accuracy)


    def select_online_random_hetero(self, budget, cores, true_index):
        '''The online random selection. heterogeneous version
        Attributes:
            budget (int):
            cores (int):
            cost_filename (str):
        '''
        random.seed(1)
        np.random.seed(2)
        true_transmitter = self.transmitters[true_index]         # in online selection, there is true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        plot_data = []
        subset_to_compute = []
        cost = 0
        cost_list = []

        while cost < budget and complement_index:
            print(cost, budget)
            option = []
            for index in complement_index:
                temp_cost = self.sensors[index].cost
                if cost + temp_cost <= budget:
                    option.append(index)
            if not option:
                break
            select = random.choice(option)
            ordered_insert(subset_index, select)
            subset_to_compute.append(copy.deepcopy(subset_index))
            complement_index.remove(select)
            cost += self.sensors[select].cost
            cost_list.append(cost)

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)

        for cost, result in zip(cost_list, subset_results):
            plot_data.append([str(result[0]), cost, result[1]])

        return plot_data


    def select_online_nearest(self, budget, cores, true_index):
        '''Online selection using the updated prior information by choosing the 'nearest' sensor
        Attributes:
            budget (int):
            cores (int):
        '''
        self.set_priori()
        plot_data = []
        random.seed(1)
        np.random.seed(2)
        true_transmitter = self.transmitters[true_index]         # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)

        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
            i += 1
        subset_index = [first_index]
        self.update_hypothesis(true_transmitter, subset_index)  # update the priori based on the first sensor
        self.print_subset(subset_index)
        self.print_grid(self.grid_priori)
        subset_to_compute = [copy.deepcopy(subset_index)]
        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)
        cost = 1

        while cost < budget and complement_index:
            distances = self.nearest_weighted_distance(complement_index)

            min_distances = np.argwhere(distances == np.amin(distances))  # there could be multiple min distances
            select = random.choice(min_distances)[0]
            index_nearest = complement_index[select]

            ordered_insert(subset_index, index_nearest)
            complement_index.remove(index_nearest)
            subset_to_compute.append(copy.deepcopy(subset_index))
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += 1

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)
        print(subset_results)
        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data


    def nearest_weighted_distance(self, complement_index):
        '''Compute the weighted distance according to the priori distribution for every sensor in
           the complement index list and return the all the distances
        Attributes:
            complement_index (list)
        Return:
            (np.ndarray) - index
        '''
        distances = []
        for index in complement_index:
            sensor = self.sensors[index]
            weighted_distance = 0
            for transmitter in self.transmitters:
                tran_x, tran_y = transmitter.x, transmitter.y
                weighted_distance += distance.euclidean([sensor.x, sensor.y], [tran_x, tran_y]) * self.grid_priori[tran_x][tran_y]
            distances.append(weighted_distance)
        return np.array(distances)


    def select_online_nearest_hetero(self, budget, cores, true_index):
        '''Online selection using the updated prior information by choosing the 'nearest' sensor
        Attributes:
            budget (int):
            cores (int):
        '''
        self.set_priori()
        '''
        energy = pd.read_csv('data/energy.txt', header=None)
        size = energy[1].count()
        i = 0
        for sensor in self.sensors:
            setattr(self.sensors.get(sensor), 'cost', energy[1][i%size])
            i += 1
        '''
        plot_data = []
        random.seed(1)
        np.random.seed(2)
        true_transmitter = self.transmitters[true_index]         # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)

        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor.x, sensor.y])
            if temp_dis < min_dis:
                min_dis = temp_dis
                first_index = i
            i += 1
        subset_index = [first_index]

        self.update_hypothesis(true_transmitter, subset_index)  # update the priori based on the first sensor
        self.print_grid(self.grid_priori)

        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)
        cost = self.sensors[first_index].cost
        subset_to_compute = [copy.deepcopy(subset_index)]
        cost_list = [cost]

        while cost < budget and complement_index:
            print(cost, budget)
            distances = self.nearest_weighted_distance(complement_index)
            min_dist_cost = distances[0] * self.sensors[complement_index[0]].cost
            best_candidate = complement_index[0]
            for dist, sen_index in zip(distances, complement_index):
                sen_cost = self.sensors[sen_index].cost
                dist_cost = dist * sen_cost
                if dist_cost < min_dist_cost:
                    min_dist_cost = dist_cost
                    best_candidate = sen_index

            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            subset_to_compute.append(copy.deepcopy(subset_index))
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += self.sensors[best_candidate].cost
            cost_list.append(cost)

        print(len(subset_to_compute), subset_to_compute)
        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for cost, result in zip(cost_list, subset_results):
            plot_data.append((str(result[0]), cost, result[1]))

        return plot_data


    def transmitters_to_array(self):
        '''transform the transmitter objects to numpy array, for the sake of CUDA
        '''
        mylist = []
        for transmitter in self.transmitters:
            templist = []
            for mean in transmitter.mean_vec:
                templist.append(mean)
            mylist.append(templist)
        self.meanvec_array = np.array(mylist)


    def o_t_approx_host(self, subset_index):
        '''host code for o_t_approx.
        Attributes:
            subset_index (np.ndarray, n=1): index of some sensors
        '''
        n_h = len(self.transmitters)   # number of hypotheses/transmitters
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)           # inverse
        d_meanvec_array = cuda.to_device(self.meanvec_array)
        d_subset_index = cuda.to_device(subset_index)
        d_sub_cov_inv = cuda.to_device(sub_cov_inv)
        d_results = cuda.device_array((n_h, n_h), np.float64)

        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        priori = self.grid_priori[0][0]                    # priori is uniform, equal everywhere

        o_t_approx_kernal[blockspergrid, threadsperblock](d_meanvec_array, d_subset_index, d_sub_cov_inv, priori, d_results)

        results = d_results.copy_to_host()
        return 1 - results.sum()


    def o_t_approx_host_2(self, subset_index, cuda_kernal):
        '''host code for o_t_approx.
        Attributes:
            subset_index (np.ndarray, n=1): index of some sensors
        '''
        n_h = len(self.transmitters)   # number of hypotheses/transmitters
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)           # inverse
        d_meanvec_array = cuda.to_device(self.meanvec_array)
        d_subset_index = cuda.to_device(subset_index)
        d_sub_cov_inv = cuda.to_device(sub_cov_inv)
        d_results = cuda.device_array((n_h, n_h), np.float64)

        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)
        priori = self.grid_priori[0][0]                    # priori is uniform, equal everywhere

        cuda_kernal[blockspergrid, threadsperblock](d_meanvec_array, d_subset_index, d_sub_cov_inv, priori, d_results)

        results = d_results.copy_to_host()
        #print_results(results)
        return 1 - results.sum()


    #@profile
    def o_t_host(self, subset_index):
        '''host code for o_t.
        Attributes:
            subset_index (np.ndarray, n=1): index of some sensors
        '''
        n_h = len(self.transmitters)   # number of hypotheses/transmitters
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)           # inverse
        d_meanvec_array = cuda.to_device(self.meanvec_array)
        d_subset_index = cuda.to_device(subset_index)
        d_sub_cov_inv = cuda.to_device(sub_cov_inv)
        d_results = cuda.device_array((n_h, n_h), np.float64)

        threadsperblock = (self.TPB, self.TPB)
        blockspergrid_x = math.ceil(n_h/threadsperblock[0])
        blockspergrid_y = math.ceil(n_h/threadsperblock[1])
        blockspergrid = (blockspergrid_x, blockspergrid_y)

        o_t_kernal[blockspergrid, threadsperblock](d_meanvec_array, d_subset_index, d_sub_cov_inv, d_results)

        results = d_results.copy_to_host()
        return np.sum(results.prod(axis=1)*self.grid_priori[0][0])


def new_data():
    '''Change config.json file, i.e. grid len and sensor number, then generate new data.
    '''
    selectsensor = SelectSensor('config.json')

    selectsensor.init_random_sensors()
    selectsensor.save_sensor('data/sensor.txt')

    selectsensor.read_init_sensor('data/sensor.txt')
    selectsensor.save_mean_std('data/mean_std.txt')

    selectsensor.read_init_sensor('data/sensor.txt')
    selectsensor.read_mean_std('data/mean_std.txt')
    selectsensor.generate_data('data/artificial_samples.csv')


def main():
    '''main
    '''
    selectsensor = SelectSensor('config.json')

    #real data
    #selectsensor.init_from_real_data('data32/homogeneous/cov', 'data32/homogeneous/sensors', 'data32/homogeneous/hypothesis')
    #print('[302, 584]', selectsensor.o_t_approx_host(np.array([302, 584])))  # two different subset generating the same o_t_approx
    #print('[383, 584]', selectsensor.o_t_approx_host(np.array([383, 584])))
    #selectsensor.init_from_real_data('data2/heterogeneous/cov', 'data2/heterogeneous/sensors', 'data2/heterogeneous/hypothesis')
    #print('o_t_approx:', selectsensor.o_t_approx_host_2(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]), o_t_approx_kernal), '\n\n')
    #print('o_t_approx_dist:', selectsensor.o_t_approx_host_2(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]), o_t_approx_dist_kernal))
    #start = time.time()
    #plots.figure_1a(selectsensor, o_t_approx_dist_kernal)
    #print('time:', time.time()-start)
    #selectsensor.init_from_real_data('data2/heterogeneous/cov', 'data2/heterogeneous/sensors', 'data2/heterogeneous/hypothesis')
    #plots.figure_1b(selectsensor)

    #fake data
    selectsensor.read_init_sensor('data/sensor.txt')
    selectsensor.read_mean_std('data/mean_std.txt')
    selectsensor.compute_multivariant_gaussian('data/artificial_samples.csv')

    #plot_data = selectsensor.select_online_greedy(3, 250)
    #plot_data = selectsensor.select_online_greedy_2(3, 250)
    start = time.time()
    plot_data = selectsensor.select_online_greedy_p_2(3, 4, 250)
    print('time:', time.time()-start)
    plots.save_data(plot_data, 'plot_data16/Online_Greedy_v2_.csv')

    #print('cpu  o_t:', selectsensor.o_t([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    #print('cuda o_t:', selectsensor.o_t_host(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11])))
    #for _ in range(100):
    #    selectsensor.o_t_host(np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))

    #print('cpu :', selectsensor.o_t_approximate([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    #print('cuda o_t_approx', selectsensor.o_t_approx_host(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])))
    #for _ in range(10000):
    #    selectsensor.o_t_approx_host(np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]))
        #print('cuda o_t_approx', selectsensor.o_t_approx_host(np.array([1, 2, 3, 4, 5, 7])))
    #print('cuda o_t_approx', selectsensor.o_t_approx_host(np.array([1, 2, 3, 4, 5, 7])))

    #start = time.time()
    #for _ in range(5000):
    #    selectsensor.o_t_approx_host(np.array([1, 2, 3, 4, 5, 7]))
    #print('time for one o_t_approx:', (time.time() - start)/5000.)

    #plots.figure_1a(selectsensor)
    #plots.figure_1a(selectsensor)
    #plot_data = selectsensor.select_offline_greedy_p(10, 4)
    #plots.save_data_offline_greedy(plot_data, 'plot_data15/Offline_Greedy.csv')

if __name__ == '__main__':
    #new_data()
    main()
