'''
Select sensor and detect transmitter
'''
import random
import math
import copy
import time
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from scipy.stats import norm
from joblib import Parallel, delayed
from sensor import Sensor
from transmitter import Transmitter
from utility import read_config
from utility import ordered_insert
import plots


class SelectSensor:
    '''Near-optimal low-cost sensor selection

    Attributes:
        config (json):       configurations - settings and parameters
        sen_num (int):       the number of sensors
        grid_len (int):      the length of the grid
        grid_priori (np.ndarray):    the element is priori probability of hypothesis - transmitter
        grid_posterior (np.ndarray): the element is posterior probability of hypothesis - transmitter
        transmitters (list): a list of Transmitter
        sensors (dict):      a dictionary of Sensor. less than 10% the # of transmitter
        data (ndarray):      a 2D array of observation data
        covariance (list):   a 2D list of covariance. each data share a same covariance matrix
        mean_stds (dict):    assume sigal between a transmitter-sensor pair is normal distributed
        subset (dict):       a subset of all sensors
        subset_index (list): the linear index of sensor in self.sensors
    '''
    def __init__(self, filename):
        self.config = read_config(filename)
        self.sen_num = int(self.config["sensor_number"])
        self.grid_len = int(self.config["grid_length"])
        self.grid_priori = np.zeros(0)
        self.grid_posterior = np.zeros(0)
        self.transmitters = []
        self.sensors = {}
        self.data = np.zeros(0)
        self.covariance = []
        self.init_transmitters()
        self.set_priori()
        self.means_stds = {}
        self.subset = {}
        self.subset_index = []


    def set_priori(self):
        '''Set priori distribution - uniform distribution
        '''
        uniform = 1./(self.grid_len * self.grid_len)
        self.grid_priori = np.full((self.grid_len, self.grid_len), uniform)
        self.grid_posterior = self.grid_priori


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
            if self.sensors.get((x, y)): # a sensor exists at (x, y)
                continue
            else:                        # no sensor exists at (x,y)
                self.sensors[(x, y)] = Sensor(x, y, random.uniform(noise_l, noise_h))  # the noise is here
                i += 1


    def save_sensor(self, filename):
        '''Save location of sensors
        '''
        with open(filename, 'w') as f:
            for key in self.sensors:
                f.write(self.sensors[key].output())


    def read_init_sensor(self, filename):
        '''Read location of sensors and init the sensors
        Attributes:
            filename (str)
        '''
        self.sensors = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y, std = int(line[0]), int(line[1]), float(line[2])
                self.sensors[(x, y)] = Sensor(x, y, std)


    def save_mean_std(self, filename):
        '''Save the mean and std of each transmitter-sensor pair.
           Mean is computed by f(x) = 100 - 30*math.log(2*dist)
        Attributes:
            filename (str)
        '''
        with open(filename, 'w') as f:
            for transmitter in self.transmitters:
                tran_x, tran_y = transmitter.x, transmitter.y
                for key in self.sensors:
                    sen_x, sen_y, std = self.sensors[key].x, self.sensors[key].y, self.sensors[key].std
                    dist = distance.euclidean([sen_x, sen_y], [tran_x, tran_y])
                    dist = 0.5 if dist < 1e-2 else dist  # in case distance is zero
                    mean = 100 - 22.2*math.log(2*dist)
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
        while i < 1000:                  # sample 1000 times for a single transmitter
            one_transmitter = []
            for sensor in self.sensors:  # for each transmitter, send signal to all sensors
                sen_x, sen_y = sensor[0], sensor[1]
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
        print('Computed covariance!')                 # assume all transmitters share the same covariance

        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            transmitter.mean_vec = []
            for sensor in self.sensors:
                sen_x, sen_y = sensor[0], sensor[1]
                mean_std = self.means_stds.get((tran_x, tran_y, sen_x, sen_y))
                transmitter.mean_vec.append(mean_std[0])
            setattr(transmitter, 'multivariant_gaussian', multivariate_normal(mean=transmitter.mean_vec, cov=self.covariance))


    def no_selection(self):
        '''The subset is all the sensors
        '''
        self.subset = copy.deepcopy(self.sensors)


    def update_subset(self, subset_index):
        '''Given a list of sensor indexes, which represents a subset of sensors, update self.subset
        Attributes:
            subset_index (list): a list of sensor indexes. guarantee sorted
        '''
        self.subset_index = subset_index
        sensor_list = list(self.sensors)           # list of sensors' key
        for index in self.subset_index:
            self.subset[sensor_list[index]] = self.sensors.get(sensor_list[index])


    def update_transmitters(self):
        '''Given a subset of sensors' index,
           update each transmitter's mean vector sub and multivariate gaussian function
        '''
        for transmitter in self.transmitters:
            transmitter.mean_vec_sub = []
            for index in self.subset_index:
                transmitter.mean_vec_sub.append(transmitter.mean_vec[index])
            new_cov = []
            for x in self.subset_index:
                row = []
                for y in self.subset_index:
                    row.append(self.covariance[x][y])
                new_cov.append(row)
            transmitter.multivariant_gaussian = multivariate_normal(mean=transmitter.mean_vec_sub, cov=new_cov)


    def select_offline_random(self, number, cores):
        '''Select a subset of sensors randomly
        Attributes:
            number (int): number of sensors to be randomly selected
            cores (int): number of cores for parallelization
        Return:
            (list): results to be plotted. each element is (str, int, float),
                    where str is the list of selected sensors, int is # of sensor, float is O_T
        '''
        self.subset = {}
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

        #self.update_subset(subset_index)
        #self.update_transmitters()
        return plot_data


    def inner_random(self, subset_index):
        '''Inner loop for random
        '''
        o_t = self.o_t(subset_index)
        return (subset_index, o_t)


    def covariance_sub(self, subset_index):
        '''Given a list of index of sensors, return the sub covariance matrix
        Attributes:
            subset_index (index): list of index of sensors. should be sorted.
        Return:
            (list): a 2D sub covariance matrix
        '''
        sub_cov = []
        for x in subset_index:
            row = []
            for y in subset_index:
                row.append(self.covariance[x][y])
            sub_cov.append(row)
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
            pj_pi = np.array(transmitter_j.mean_vec_sub) - np.array(transmitter_i.mean_vec_sub)
            prob_i.append(1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
        product = 1
        for i in prob_i:
            product *= i
        return product*self.grid_priori[i_x][i_y]


    def o_t(self, subset_index):
        '''Given a subset of sensors T, compute the O_T
        Attributes:
            subset_index (list): a subset of sensors T, guarantee sorted
        Return O_T
        '''
        if not subset_index:  # empty sequence are false
            return 0
        prob_error = []
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse

        for transmitter_i in self.transmitters:
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = []
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = np.array(transmitter_j.mean_vec_sub) - np.array(transmitter_i.mean_vec_sub)
                prob_i.append(1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            product = 1
            for i in prob_i:
                product *= i
            prob_error.append(product * self.grid_priori[i_x][i_y])
        o_t = 0
        for i in prob_error:
            o_t += i
        return o_t


    def o_t_approximate(self, subset_index):
        '''Not the accurate O_T, but apprioximating O_T. So that we have a good propertiy of submodular
        Attributes:
            subset_index (list): a subset of sensors T, needs guarantee sorted
        '''
        if not subset_index:  # empty sequence are false
            return -99999999999.
        prob_error = []
        sub_cov = self.covariance_sub(subset_index)
        sub_cov_inv = np.linalg.inv(sub_cov)        # inverse

        for transmitter_i in self.transmitters:
            i_x, i_y = transmitter_i.x, transmitter_i.y
            transmitter_i.set_mean_vec_sub(subset_index)
            prob_i = []
            for transmitter_j in self.transmitters:
                j_x, j_y = transmitter_j.x, transmitter_j.y
                if i_x == j_x and i_y == j_y:
                    continue
                transmitter_j.set_mean_vec_sub(subset_index)
                pj_pi = np.array(transmitter_j.mean_vec_sub) - np.array(transmitter_i.mean_vec_sub)
                prob_i.append(norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            summation = 0
            for i in prob_i:
                summation += i
            prob_error.append(summation * self.grid_priori[i_x][i_y])
        error = 0
        for i in prob_error:
            error += i
        return 1 - error


    def select_offline_greedy_p(self, budget, cores, latency=False):
        '''(Parallel version) Select a subset of sensors greedily. offline + homo version
        Attributes:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        plot_data = []

        sensor_list = list(self.sensors)                    # list of sensors' key
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        start = time.time()
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
            cost += self.sensors.get(sensor_list[best_candidate]).cost
            if latency:
                plot_data.append([str(subset_index), len(subset_index), time.time()-start])  # Y value is latency
            else:
                o_t_real = self.o_t(subset_index)
                plot_data.append([str(subset_index), len(subset_index), o_t_real]) # TODO optimization can be done here. like the hetero version

        self.update_subset(subset_index)
        self.update_transmitters()

        return plot_data


    def inner_greedy(self, subset_index, candidate):
        '''Inner loop for selecting candidates
        Attributes:
            subset_index (list):
            candidate (int):
        Return:
            (tuple): (index, o_t_approx, new subset_index)
        '''
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)     # guarantee subset_index always be sorted here
        o_t = self.o_t_approximate(subset_index2)
        return (candidate, o_t, subset_index2)


    def select_offline_greedy(self, budget):
        '''Select a subset of sensors greedily. offline + homo version
        Attributes:
            budget (int): budget constraint
        Return:
            (list): an element is [str, int, float],
                    where str is the list of subset_index, int is # of sensors, float is O_T
        '''
        sensor_list = list(self.sensors)                    # list of sensors' key
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
            cost += self.sensors.get(sensor_list[best_candidate]).cost

        self.update_subset(subset_index)
        self.update_transmitters()

        return plot_data


    def select_offline_random_hetero(self, budget, cores, cost_filename):
        '''Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two

        Attributes:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
        random.seed(0)    # though algorithm is random, the results are the same every time

        energy = pd.read_csv(cost_filename, header=None)
        size = energy[1].count()
        i = 0
        for sensor in self.sensors:
            setattr(self.sensors.get(sensor), 'cost', energy[1][i%size])
            i += 1

        sensor_list = list(self.sensors)                    # list of sensors' key
        self.subset = {}
        subset_index = []
        plot_data = []
        sequence = [i for i in range(self.sen_num)]
        cost = 0
        subset_to_compute = []
        for budget_i in range(1, budget):
            while cost < budget_i:
                option = []
                for index in sequence:
                    temp_cost = self.sensors.get(sensor_list[index]).cost
                    if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                        option.append(index)
                if not option:                      # if there are no sensors that can be selected, then break
                    break
                select = random.choice(option)
                ordered_insert(subset_index, select)
                subset_to_compute.append(copy.deepcopy(subset_index))
                sequence.remove(select)
                cost += self.sensors.get(sensor_list[select]).cost

        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_random)(subset_index) for subset_index in subset_to_compute)

        for result in subset_results:
            plot_data.append((str(result[0]), len(result[0]), result[1]))

        #self.update_subset(subset_index)
        #self.update_transmitters()
        return plot_data


    def select_offline_greedy_hetero(self, budget, cores, cost_filename):
        '''Offline selection when the sensors are heterogeneous
           Two pass method: first do a homo pass, then do a hetero pass, choose the best of the two

        Attributes:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
        energy = pd.read_csv(cost_filename, header=None)
        size = energy[1].count()
        i = 0
        for sensor in self.sensors:
            setattr(self.sensors.get(sensor), 'cost', energy[1][i%size])
            i += 1

        sensor_list = list(self.sensors)                    # list of sensors' key
        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        maximum = 0
        first_pass_plot_data = []
        while cost < budget and complement_index:
            option = []
            for index in complement_index:
                temp_cost = self.sensors.get(sensor_list[index]).cost
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
            cost += self.sensors.get(sensor_list[best_candidate]).cost
            first_pass_plot_data.append([copy.deepcopy(subset_index), len(subset_index), 0])           # Y value is real o_t
            print(subset_index, maximum, cost)

        print('end of the first homo pass and start of the second hetero pass')

        cost = 0                                            # |T| in the paper
        subset_index = []                                   # T   in the paper
        complement_index = [i for i in range(self.sen_num)] # S\T in the paper
        maximum = 0
        base_ot = 1 - 0.5*len(self.transmitters)            # O_T from the previous iteration
        second_pass_plot_data = []
        while cost < budget and complement_index:
            option = []
            for index in complement_index:
                temp_cost = self.sensors.get(sensor_list[index]).cost
                if cost + temp_cost <= budget:  # a sensor can be selected if adding its cost is under budget
                    option.append(index)
            if not option:
                break

            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy)(subset_index, candidate) for candidate in option)

            best_candidate = candidate_results[0][0]                       # an element of candidate_results is a tuple - (int, float, list)
            cost_of_candiate = self.sensors.get(sensor_list[best_candidate]).cost
            new_base_ot = candidate_results[0][1]
            maximum = (candidate_results[0][1]-base_ot)/cost_of_candiate   # where int is the candidate, float is the O_T, list is the subset_list with new candidate
            for candidate in candidate_results:
                incre = candidate[1] - base_ot
                cost_of_candiate = self.sensors.get(sensor_list[candidate[0]]).cost
                incre_cost = incre/cost_of_candiate     # increment of O_T devided by cost
                print(candidate[2], candidate[1], incre, cost_of_candiate, incre_cost)
                if incre_cost > maximum:
                    best_candidate = candidate[0]
                    maximum = incre_cost
                    new_base_ot = candidate[1]
            base_ot = new_base_ot
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            complement_index.remove(best_candidate)
            cost += self.sensors.get(sensor_list[best_candidate]).cost
            second_pass_plot_data.append([copy.deepcopy(subset_index), len(subset_index), 0])           # Y value is real o_t
            print(subset_index, base_ot, cost)

        first_pass = []
        for data in first_pass_plot_data:
            first_pass.append(data[0])
        second_pass = []
        for data in second_pass_plot_data:
            second_pass.append(data[0])

        first_pass_o_ts = Parallel(n_jobs=len(first_pass_plot_data))(delayed(self.inner_greedy_hetero)(subset_index) for subset_index in first_pass)
        second_pass_o_ts = Parallel(n_jobs=len(second_pass_plot_data))(delayed(self.inner_greedy_hetero)(subset_index) for subset_index in second_pass)

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


    def inner_greedy_hetero(self, subset_index):
        '''Compute o_t
        '''
        o_t = self.o_t(subset_index)
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
            temp_dis = distance.euclidean([center[0], center[1]], [sensor[0], sensor[1]])
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


    def select_offline_coverage_hetero(self, budget, cores, cost_filename):
        '''A coverage-based baseline algorithm (heterogeneous version)
        '''
        random.seed(0)
        sensor_list = list(self.sensors)                    # list of sensors' key
        energy = pd.read_csv(cost_filename, header=None)  # load the energy cost
        size = energy[1].count()
        i = 0
        for sensor in self.sensors:
            setattr(self.sensors.get(sensor), 'cost', energy[1][i%size])
            i += 1

        center = (int(self.grid_len/2), int(self.grid_len/2))
        min_dis = 99999
        first_index, i = 0, 0
        first_sensor = None
        for sensor in self.sensors:        # select the first sensor that is closest to the center of the grid
            temp_dis = distance.euclidean([center[0], center[1]], [sensor[0], sensor[1]])
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
        cost = self.sensors.get(sensor_list[first_index]).cost

        while cost < budget and complement_index:
            min_overlap_cost = 99999   # to minimize overlap*cost
            best_candidate = []
            best_sensor = []
            for candidate in complement_index:
                sensor = self.index_to_sensor(candidate)
                overlap = self.compute_overlap(coverage, sensor, radius)
                temp_cost = self.sensors.get(sensor_list[candidate]).cost
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
            cost += self.sensors.get(sensor_list[best_candidate[choose]]).cost

        print(len(subset_to_compute), subset_to_compute)
        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_random)(subset_index) for subset_index in subset_to_compute)

        plot_data = []
        for result in subset_results:
            plot_data.append((str(result[0]), len(result[0]), result[1]))

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
            transmitter_i = self.transmitters[(first_sensor[0] - i)*self.grid_len + first_sensor[1]] # 2D index --> 1D index
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
                pj_pi = np.array(transmitter_j.mean_vec_sub) - np.array(transmitter_i.mean_vec_sub)
                prob_i.append(1 - norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))))
            product = 1
            for prob in prob_i:
                product *= prob
            if product > 0.0001:     # set threshold
                radius = i
            else:
                break
        return radius


    def compute_overlap(self, coverage, sensor, radius):
        '''Compute the overlap between selected sensors and the new sensor
        '''
        x_low = sensor[0] - radius if sensor[0] - radius >= 0 else 0
        x_high = sensor[0] + radius if sensor[0] + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor[1] - radius if sensor[1] - radius >= 0 else 0
        y_high = sensor[1] + radius if sensor[1] + radius <= self.grid_len-1 else self.grid_len-1

        overlap = 0
        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high):
                if distance.euclidean([x, y], [sensor[0], sensor[1]]) <= radius:
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
            sensor (tuple): (x, y)
            radius (int): radius of a sensor
        '''
        x_low = sensor[0] - radius if sensor[0] - radius >= 0 else 0
        x_high = sensor[0] + radius if sensor[0] + radius <= self.grid_len-1 else self.grid_len-1
        y_low = sensor[1] - radius if sensor[1] - radius >= 0 else 0
        y_high = sensor[1] + radius if sensor[1] + radius <= self.grid_len-1 else self.grid_len-1

        for x in range(x_low, x_high+1):
            for y in range(y_low, y_high+1):
                if distance.euclidean([x, y], [sensor[0], sensor[1]]) <= radius:
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
                    sen_x, sen_y = sensor[0], sensor[1]
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


    def select_online_random(self, budget):
        '''The online random selection
        Attributes:
            budget (int)
        '''
        random.seed(1)
        np.random.seed(2)
        rand = random.randint(0, self.grid_len*self.grid_len-1)
        true_transmitter = self.transmitters[rand]         # in online selection, there is true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        plot_data = []
        self.print_priori()
        cost = 0

        while cost < budget and complement_index:
            maximum = self.mutual_information(subset_index, true_transmitter)
            best_candidate = complement_index[0]
            for candidate in complement_index:
                ordered_insert(subset_index, candidate)
                temp = self.mutual_information(subset_index, true_transmitter)
                print(subset_index, 'MI =', temp)
                if temp > maximum:
                    maximum = temp
                    best_candidate = candidate
                subset_index.remove(candidate)
            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            cost += 1
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_priori()
            print('\n')
        return plot_data


    def select_online_greedy(self, budget):
        '''The online greedy selection
        Attributes:
            budget (int)
        '''
        random.seed(1)
        np.random.seed(2)
        rand = random.randint(0, self.grid_len*self.grid_len-1)
        true_transmitter = self.transmitters[rand]         # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        plot_data = []
        self.print_priori()
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
            cost += 1
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_priori()
            print('\n')
        return plot_data


    def print_subset(self, subset_index):
        '''Print the subset_index and its 2D location
        Attriubtes:
            subset_index (list)
        '''
        print(subset_index, end=' ')
        subset_list = list(self.sensors)
        print('[', end=' ')
        for index in subset_index:
            print(subset_list[index], end=' ')
        print(']')


    def print_priori(self):
        '''Print priori matrix
        '''
        size = len(self.grid_priori)
        for i in range(size):
            print('[', end=' ')
            for j in range(size):
                print('%.5f' % self.grid_priori[i][j], end=' ')
            print(']')


    def update_hypothesis(self, true_transmitter, subset_index):
        '''Use Bayes formula to update P(hypothesis): form prior to posterior
           After we add a new sensor and get a larger subset, the larger subset begins to observe data from true transmitter
        Attributes:
            true_transmitter (Transmitter)
            subset_index (list)
        '''
        self.subset_index = subset_index
        self.update_transmitters()
        true_x, true_y = true_transmitter.x, true_transmitter.y
        data = []                          # the true transmitter generate some data
        sensor_list = list(self.sensors)
        for index in subset_index:
            sensor = sensor_list[index]
            mean, std = self.means_stds.get((true_x, true_y, sensor[0], sensor[1]))
            data.append(np.random.normal(mean, std))
        for trans in self.transmitters:
            likelihood = trans.multivariant_gaussian.pdf(data)
            self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
        denominator = self.grid_posterior.sum()
        try:
            self.grid_posterior = self.grid_posterior/denominator
            self.grid_priori = self.grid_posterior   # the posterior in this iteration will be the prior in the next iteration
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
        if size > number:                 # remove redundant hypotheses
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
        sensor_list = list(self.sensors)
        sensor_observe = []                  # R2 in the paper
        for hypothesis in true_hypotheses:   # R1 in the paper
            true_transmitter = self.transmitters[hypothesis]
            true_x, true_y = true_transmitter.x, true_transmitter.y
            data = []                        # data generated from true transmitter
            for index in subset_index:
                sensor = sensor_list[index]
                mean, std = self.means_stds.get((true_x, true_y, sensor[0], sensor[1]))
                data.append(np.random.normal(mean, std))
            for trans in self.transmitters:
                likelihood = trans.multivariant_gaussian.pdf(data)
                self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y]
            r2 = np.argmax(self.grid_posterior)
            sensor_observe.append(r2)
        
        


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


def figure_1a(selectsensor):
    '''Y - Probability of error
       X - # of sensor
       Homogeneous
       Algorithm - Offline greedy and offline random
    '''
    #plot_data = selectsensor.select_offline_greedy_p(20, 20)
    #plots.save_data(plot_data, 'plot_data2/Offline_Greedy_30.csv')

    #plot_data = selectsensor.select_offline_random(40, 20)
    #plots.save_data(plot_data, 'plot_data2/Offline_Random_30.csv')

    plot_data = selectsensor.select_offline_coverage(20, 4)
    plots.save_data(plot_data, 'plot_data2/Offline_Coverage_15.csv')


def figure_1b(selectsensor):
    '''Y - Probability of error
       X - Total budget
       Heterogeneous
       Algorithm - Offline greedy and offline random
    '''

    plot_data = selectsensor.select_offline_random_hetero(15, 40, 'data/energy.txt')
    plots.save_data(plot_data, 'plot_data2/Offline_Random_30_hetero.csv')

    #plot_data = selectsensor.select_offline_greedy_hetero(15, 40, 'data/energy.txt')
    #plots.save_data(plot_data, 'plot_data2/Offline_Greedy_30_hetero.csv')

    plot_data = selectsensor.select_offline_coverage_hetero(15, 40, 'data/energy.txt')
    plots.save_data(plot_data, 'plot_data2/Offline_Coverage_30_hetero.csv')


def figure_1c(selectsensor):
    '''Y - Latency
       X - Number of sensors selected
       Algorithm - Offline greedy
    '''
    plot_data = selectsensor.select_offline_greedy_p(20, 4, latency=True)
    plots.save_data(plot_data, 'plot_data2/Latency_20.csv')

    #plot_data = selectsensor.select_offline_greedy_p(20, 4, latency=True)
    #plots.save_data(plot_data, 'plot_data2/Latency_50.csv')

    #plot_data = selectsensor.select_offline_greedy_p(20, 4, latency=True)
    #plots.save_data(plot_data, 'plot_data2/Latency_100.csv')


def main():
    '''main
    '''

    selectsensor = SelectSensor('config.json')

    selectsensor.read_init_sensor('data/sensor.txt')
    selectsensor.read_mean_std('data/mean_std.txt')
    selectsensor.compute_multivariant_gaussian('data/artificial_samples.csv')

    plot_data = selectsensor.select_online_greedy(5)
    plots.save_data(plot_data, 'plot_data2/Online_Greedy_15.csv')

    #figure_1b(selectsensor)

    #plot_data = selectsensor.select_offline_greedy(10)
    #plots.save_data(plot_data, 'plot_data2/test_of_approx.csv')
    #figure_1b(selectsensor)

    #selectsensor.no_selection()

    #plot_data = selectsensor.select_offline_greedy_p(20, 4)

    #plot_data = selectsensor.select_offline_random(20)
    #plot_data = selectsensor.select_offline_greedy_p(20, 40)

    #print('error ', selectsensor.test_error())


if __name__ == '__main__':
    #new_data()
    main()
