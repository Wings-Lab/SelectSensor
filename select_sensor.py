'''
Select sensor and detect transmitter
'''

import random
import math
import copy
import time
import os
import numpy as np
import pandas as pd
from numba import cuda
from scipy.spatial import distance
from scipy.stats import multivariate_normal, norm
from joblib import Parallel, delayed, dump, load
from sensor import Sensor
from transmitter import Transmitter
from utility import read_config, ordered_insert#, print_results
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
        self.discre_x_file = self.config["discretized_x_file"]
        self.grid_priori = np.zeros(0)
        self.grid_posterior = np.zeros(0)
        self.transmitters = []
        self.sensors = []
        self.data = np.zeros(0)
        self.covariance = np.zeros(0)
        self.init_transmitters()
        self.set_priori()
        self.means = np.zeros(0)
        self.stds = np.zeros(0)
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

        self.means = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        self.stds = np.zeros((self.grid_len * self.grid_len, len(self.sensors)))
        with open(hypothesis_file, 'r') as f:
            lines = f.readlines()
            count = 0
            for line in lines:
                line = line.split(' ')
                tran_x, tran_y = int(line[0]), int(line[1])
                #sen_x, sen_y = int(line[2]), int(line[3])
                mean, std = float(line[4]), float(line[5])
                self.means[tran_x*self.grid_len + tran_y, count] = mean  # count equals to the index of the sensors
                self.stds[tran_x*self.grid_len + tran_y, count] = std
                count = (count + 1) % len(self.sensors)

        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            mean_vec = [0] * len(self.sensors)
            for sensor in self.sensors:
                mean = self.means[self.grid_len*tran_x + tran_y, sensor.index]
                mean_vec[sensor.index] = mean
            transmitter.mean_vec = np.array(mean_vec)
        #self.transmitters_to_array() # for GPU
        #del self.means_stds  # in 64*64 grid offline case, need to delete means_stds. otherwise exceed 4GB limit of joblib
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
        self.transmitters = [0] * self.grid_len * self.grid_len
        for i in range(self.grid_len):
            for j in range(self.grid_len):
                transmitter = Transmitter(i, j)
                setattr(transmitter, 'hypothesis', i*self.grid_len + j)
                self.transmitters[i*self.grid_len + j] = transmitter


    def update_subset(self, subset_index):
        '''Given a list of sensor indexes, which represents a subset of sensors, update self.subset
        Args:
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
        Args:
            subset_index (list)
        '''
        for transmitter in self.transmitters:
            transmitter.set_mean_vec_sub(subset_index)


    def select_offline_random(self, number, cores):
        '''Select a subset of sensors randomly
        Args:
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
        Args:
            subset_index (list): list of index of sensors. should be sorted.
        Return:
            (np.ndarray): a 2D sub covariance matrix
        '''
        sub_cov = self.covariance[np.ix_(subset_index, subset_index)]
        return sub_cov


    def o_t_p(self, subset_index, cores):
        '''(Parallelized version of o_t function) Given a subset of sensors T, compute the O_T
        Args:
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
        Args:
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
        Args:
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
        Args:
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
        Args:
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
        Args:
            budget (int): budget constraint
            cores (int): number of cores for parallelzation
            cuda_kernal (cuda_kernals.o_t_approx_kernal or o_t_approx_dist_kernal): the O_{aux} in the paper
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
            best_candidate = complement_sensors[0].index   # init as the first sensor
            best_sensor = complement_sensors[0]
            complement_sensors.sort()   # sorting the gain descendingly
            new_base_ot_approx = 0
            #for sensor in complement_sensors:
            #    print((sensor.index, sensor.gain_up_bound), end=' ')
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
                    #print('\n***LAZY!***\n', cost, (update, update_end), len(complement_sensors), '\n')
                    break
                update += cores
            base_ot_approx = new_base_ot_approx             # update the base o_t_approx for the next iteration
            print(best_candidate, subset_index, base_ot_approx, '\n')
            ordered_insert(subset_index, best_candidate)    # guarantee subset_index always be sorted here
            subset_to_compute.append(copy.deepcopy(subset_index))
            plot_data.append([len(subset_index), base_ot_approx, 0]) # don't compute real o_t now, delay to after all the subsets are selected
            complement_sensors.remove(best_sensor)
            if base_ot_approx > 0.9999999999999:
                break
            cost += 1
        print('number of o_t_approx', counter)
        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_greedy_real_ot)(subset_index) for subset_index in subset_to_compute)

        for i in range(len(subset_results)):
            plot_data[i][2] = subset_results[i]

        return plot_data


    def inner_greedy(self, subset_index, cuda_kernal, candidate):
        '''Inner loop for selecting candidates
        Args:
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
        Args:
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
        Args:
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
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
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
        Args:
            budget (int): budget we have for the heterogeneous sensors
            cores (int): number of cores for parallelization
            cost_filename (str): file that has the cost of sensors
        '''
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
        Args:
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
        Args:
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
        Args:
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


    def index_inverse(self, index):
        '''Convert 1D index into 2D index
        '''
        x = int(index/self.grid_len)
        y = index%self.grid_len
        return (x, y)


    def select_online_greedy_hetero(self, budget, cores, true_index=-1):
        '''Heterogeneous version of online greedy selection
        Args:
            budget (int): amount of budget, in the homo case, every sensor has budget=1
            cores (int): number of cores used in the parallezation
            cost_filename (str): file that has the cost of sensors
        '''
        if true_index == -1:
            random.seed()
            true_index = random.randint(0, self.grid_len * self.grid_len)
        self.set_priori()
        random.seed(1)
        np.random.seed(2)
        plot_data = []
        true_transmitter = self.transmitters[true_index]         # in online selection, there is one true transmitter somewhere
        print('\ntrue transmitter', true_transmitter)

        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        #self.print_grid(self.grid_priori)
        discretize_x = self.discretize(bin_num=200, cores=cores)
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

            candidate_results = Parallel(n_jobs=cores)(delayed(self.inner_online_greedy)(discretize_x, subset_index, candidate) \
                                                       for candidate in option)

            best_candidate = candidate_results[0][0]
            cost_of_candidate = self.sensors[best_candidate].cost
            maximum = candidate_results[0][1]/cost_of_candidate      # the metric is MI/cost
            for candidate in candidate_results:
                mi = candidate[1]
                cost_of_candidate = self.sensors[candidate[0]].cost
                mi_cost = mi/cost_of_candidate                       # the metric is MI/cost
                #print(candidate[0], mi, cost_of_candidate, mi_cost)
                if mi_cost > maximum:
                    maximum = mi_cost
                    best_candidate = candidate[0]
            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            #self.print_grid(self.grid_priori)
            cost += self.sensors[best_candidate].cost
            cost_list.append(cost)
            subset_to_compute.append(copy.deepcopy(subset_index))

        print(len(subset_to_compute), subset_to_compute)
        subset_results = Parallel(n_jobs=cores)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) \
                                                for subset_index in subset_to_compute)

        plot_data = []
        for cost, result in zip(cost_list, subset_results):
            plot_data.append((str(result[0]), cost, result[1]))

        return plot_data


    def select_online_greedy_p(self, budget, cores, true_index=-1):
        '''(Parallel version) Version 2 of online greedy selection with mutual_information version 2
        Args:
            budget (int): amount of budget, in the homo case, every sensor has budget=1
            cores (int): number of cores used in the parallezation
            true_index (int): the true transmitter
        Return:
            plot_data (list)
        '''
        if true_index == -1:
            random.seed()
            true_index = random.randint(0, self.grid_len * self.grid_len)
        self.set_priori()
        random.seed(1)
        np.random.seed(2)
        true_transmitter = self.transmitters[true_index] # in online selection, there is one true transmitter somewhere
        print('\ntrue transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        #self.print_grid(self.grid_priori)
        start = time.time()
        discretize_x = self.discretize(bin_num=100, cores=cores)
        print('discretize x', time.time() - start)
        cost = 0
        subset_to_compute = []
        mi = []                 # save the mutual informations
        start = time.time()
        while cost < budget and complement_index:
            candidate_results = Parallel(n_jobs=cores, verbose=0)(delayed(self.inner_online_greedy)(discretize_x, subset_index, candidate) \
                                                               for candidate in complement_index)

            best_candidate = candidate_results[0][0]
            maximum = candidate_results[0][1]
            for candidate in candidate_results:
                #print(candidate[2], candidate[1])
                if candidate[1] > maximum:
                    maximum = candidate[1]
                    best_candidate = candidate[0]

            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            subset_to_compute.append(copy.deepcopy(subset_index))
            print('MI = ', maximum)
            mi.append(maximum)
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            #self.print_grid(self.grid_priori)
            cost += 1
        print('MI time', time.time() - start)
        subset_results = Parallel(n_jobs=cores, verbose=60)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) \
                                                for subset_index in subset_to_compute)

        plot_data = []
        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data, mi


    def select_online_greedy_p2(self, budget, cores, true_index=-1):
        '''(Parallel version) Version 3 of online greedy selection with mutual_information version 4
        Args:
            budget (int): amount of budget, in the homo case, every sensor has budget=1
            cores (int): number of cores used in the parallezation
            true_index (int): the true transmitter
        Return:
            plot_data (list)
        '''
        if true_index == -1:
            random.seed()
            true_index = random.randint(0, self.grid_len * self.grid_len)
        self.set_priori()
        random.seed(1)
        np.random.seed(2)
        true_transmitter = self.transmitters[true_index] # in online selection, there is one true transmitter somewhere
        print('\ntrue transmitter', true_transmitter)
        subset_index = []
        complement_index = [i for i in range(self.sen_num)]
        #self.print_grid(self.grid_priori)
        start = time.time()
        #discretize_x = self.discretize2(bin_num=100, cores=cores)  # discretize2 for repeating experiment
        discretize_x = self.discretize3(bin_num=100, cores=cores)   # discretize3 for scalability test
        print('discretize x', time.time() - start)
        cost = 0
        subset_to_compute = []
        mi_list = []                 # save the mutual informations
        start = time.time()
        while cost < budget and complement_index:
            maximum = -1
            best_candidate = -1
            for candidate in complement_index:
                mi = self.mutual_information(discretize_x, candidate)
                print(candidate, mi)
                if mi > maximum:
                    maximum = mi
                    best_candidate = candidate

            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            subset_to_compute.append(copy.deepcopy(subset_index))
            print('MI = ', maximum)
            mi_list.append(maximum)
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            #self.print_grid(self.grid_priori)
            cost += 1
        print('MI time', time.time() - start)
        return  # for scalability test, don't need to compute the accuracy, so return here
        subset_results = Parallel(n_jobs=cores, verbose=60)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) \
                                                for subset_index in subset_to_compute)

        plot_data = []
        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data, mi_list


    def inner_online_greedy(self, discretize_x, subset_index, candidate):
        '''The inner loop for online greedy version 2
        Args:
            discretize_x (np.ndarray, n = 2)
            subset_index (list)
            candidate (int)
        '''
        np.random.seed(candidate)
        subset_index2 = copy.deepcopy(subset_index)
        ordered_insert(subset_index2, candidate)
        mi = self.mutual_information(discretize_x, candidate)
        return (candidate, mi, subset_index2)


    #@profile
    def select_online_greedy(self, budget, true_index):
        '''The online greedy selection version 2. Homogeneous.
        Args:
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
        discretize_x = self.discretize(200, cores)
        cost = 0

        while cost < budget and complement_index:
            maximum = self.mutual_information(discretize_x, complement_index[0])
            best_candidate = complement_index[0]
            for candidate in complement_index:
                mi = self.mutual_information(discretize_x, candidate)
                print(candidate, 'MI =', mi)
                if mi > maximum:
                    maximum = mi
                    best_candidate = candidate
            ordered_insert(subset_index, best_candidate)
            complement_index.remove(best_candidate)
            plot_data.append([str(subset_index), len(subset_index), maximum])
            print('MI = ', maximum)
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += 1
        return plot_data


    def discretize(self, bin_num, cores):
        '''Discretize the likelihood of data P(X|h) for each hypothesis
        Args:
            bin (int): bin size, discretize the X axis into bin # of bins
            cores (int): number of cores
        Return:
            (numpy.ndarray): n = 3
        '''
        min_mean = np.min(self.means)
        max_mean = np.max(self.means)
        max_std = np.max(self.stds)
        X = np.linspace(min_mean - 3*max_std, max_mean + 3*max_std, bin_num+1)

        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        data_filename_memmap = os.path.join(folder, 'data_memmap')
        dump(X, data_filename_memmap)
        X = load(data_filename_memmap, mmap_mode='r')

        output_filename_memmap = os.path.join(folder, 'output_memmap')
        discretize_x = np.zeros((len(self.sensors), self.grid_len * self.grid_len, bin_num))

        discretize_x = np.memmap(output_filename_memmap, dtype=discretize_x.dtype, shape=discretize_x.shape, mode='w+')
        Parallel(n_jobs=cores, verbose=60)(delayed(self.inner_discretize)(X, bin_num, sensor, discretize_x) for sensor in self.sensors)

        return discretize_x


    def discretize2(self, bin_num, cores):
        '''Discretize the likelihood of data P(X|h) for each hypothesis
        This version of discretization is for single core online greedy with repeating experiments
        Args:
            bin (int): bin size, discretize the X axis into bin # of bins
            cores (int): number of cores
        Return:
            (numpy.ndarray): n = 3
        '''
        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass
        discretize_x_filename = os.path.join(folder, self.discre_x_file)
        if os.path.isfile(discretize_x_filename):
            discretized_x = load(discretize_x_filename, 'r')
            return np.array(discretized_x)

        min_mean = np.min(self.means)
        max_mean = np.max(self.means)
        max_std = np.max(self.stds)
        X = np.linspace(min_mean - 3*max_std, max_mean + 3*max_std, bin_num+1)

        data_filename_memmap = os.path.join(folder, 'X_memmap')
        dump(X, data_filename_memmap)
        X = load(data_filename_memmap, mmap_mode='r')

        output_filename_memmap = os.path.join(folder, 'output_memmap')
        discretize_x = np.zeros((len(self.sensors), self.grid_len * self.grid_len, bin_num))

        discretize_x = np.memmap(output_filename_memmap, dtype=discretize_x.dtype, shape=discretize_x.shape, mode='w+')
        Parallel(n_jobs=cores, verbose=60)(delayed(self.inner_discretize)(X, bin_num, sensor, discretize_x) for sensor in self.sensors)

        discretize_x_np = np.array(discretize_x)
        dump(discretize_x_np, discretize_x_filename)
        os.remove(output_filename_memmap)
        return discretize_x_np


    def discretize3(self, bin_num, cores):
        '''Discretize the likelihood of data P(X|h) for each hypothesis
        This version of discretization is for single core online greedy with scalability test
        Args:
            bin (int): bin size, discretize the X axis into bin # of bins
            cores (int): number of cores
        Return:
            (numpy.ndarray): n = 3
        '''
        folder = './joblib_memmap'
        try:
            os.mkdir(folder)
        except FileExistsError:
            pass

        min_mean = np.min(self.means)
        max_mean = np.max(self.means)
        max_std = np.max(self.stds)
        X = np.linspace(min_mean - 3*max_std, max_mean + 3*max_std, bin_num+1)

        data_filename_memmap = os.path.join(folder, 'X_memmap')
        dump(X, data_filename_memmap)
        X = load(data_filename_memmap, mmap_mode='r')

        output_filename_memmap = os.path.join(folder, 'output_memmap')
        discretize_x = np.zeros((len(self.sensors), self.grid_len * self.grid_len, bin_num))

        discretize_x = np.memmap(output_filename_memmap, dtype=discretize_x.dtype, shape=discretize_x.shape, mode='w+')
        Parallel(n_jobs=cores, verbose=60)(delayed(self.inner_discretize)(X, bin_num, sensor, discretize_x) for sensor in self.sensors)

        discretize_x_np = np.array(discretize_x)
        return discretize_x_np


    def inner_discretize(self, X, bin_num, sensor, discretize_x):
        '''the inner loop of discretization
        Args:
            X (numpy.ndarray, n=1): the x axis
            bin_num (int): the number of bins on x axis
            sensor (Sensor)
            discretize_x ()
        '''
        for transNum in range(self.grid_len * self.grid_len):
            mean = self.means[transNum, sensor.index]
            std = self.stds[transNum, sensor.index]
            cdf = norm.cdf(X, mean, std)
            for i in range(bin_num):
                discretize_x[sensor.index, transNum, i] = cdf[i + 1] - cdf[i]


    def print_subset(self, subset_index):
        '''Print the subset_index and its 2D location
        Args:
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
           An important update from update_hypothesis to update_hypothesis_2 is that we are not using attribute transmitter.multivariant_gaussian. It saves money
        Args:
            true_transmitter (Transmitter)
            subset_index (list)
        '''
        true_x, true_y = true_transmitter.x, true_transmitter.y
        np.random.seed(true_x*self.grid_len + true_y*true_y)  # change seed here
        data = []                          # the true transmitter generate some data
        for index in subset_index:
            sensor = self.sensors[index]
            mean = self.means[self.grid_len*true_x + true_y, sensor.index]
            std = self.stds[self.grid_len*true_x + true_y, sensor.index]
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
        Args:
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
    def mutual_information_original(self, discretize_x, sensor_index):
        '''mutual information version 3
        Args:
            discretize_x (np.ndarray, n = 3): for each pair of (sensor, transmitter), discretize the gaussian distribution
            sensor_index (int): the X_e in the paper, a candidate sensor
        '''
        prob_x = []           # compute the probability of x
        x_num = len(discretize_x[0, 0])
        for i in range(x_num):
            summation = 0
            for trans in self.transmitters:
                x = trans.hypothesis // self.grid_len
                y = trans.hypothesis % self.grid_len
                summation += discretize_x[sensor_index, trans.hypothesis, i] * self.grid_priori[x, y]
            prob_x.append(summation)

        summation = 0         # compute the mutual information
        for trans in self.transmitters:
            x = trans.hypothesis // self.grid_len
            y = trans.hypothesis % self.grid_len
            for prob_xh, prob_xi in zip(discretize_x[sensor_index, trans.hypothesis], prob_x):
                if prob_xh == 0 or prob_xi == 0:
                    continue
                term = prob_xh * self.grid_priori[x, y] * math.log2(prob_xh/prob_xi)
                if not (np.isnan(term) or np.isinf(term)):
                    summation += term
        return summation

    #@profile
    def mutual_information(self, discretize_x, sensor_index):
        '''mutual information version 4 -- the optimization of version 3 by using more numpy stuff, less Python for loop
        Args:
            discretize_x (np.ndarray, n = 3): for each pair of (sensor, transmitter), discretize the gaussian distribution
            sensor_index (int): the X_e in the paper, a candidate sensor
        '''
        import warnings
        warnings.filterwarnings("ignore")                   # get rid of annoying runtime warnings such as devide by zero

        prob_x = np.zeros(len(discretize_x[0, 0]))          # compute the probability of x
        x_num = len(discretize_x[0, 0])
        for i in range(x_num):
            prob_x[i] = np.dot(discretize_x[sensor_index, :, i], self.grid_priori.flatten())

        summation = 0                                       # compute the mutual information
        for trans in self.transmitters:
            x = trans.hypothesis // self.grid_len
            y = trans.hypothesis % self.grid_len
            prob_xh = discretize_x[sensor_index, trans.hypothesis, :]
            log_term = np.log2(prob_xh / prob_x)
            log_term[log_term == np.inf] = 0
            summation_term = self.grid_priori[x, y] * log_term * prob_xh
            np.nan_to_num(summation_term, copy=False)

            summation += np.sum(summation_term)

        return summation

    #@profile
    def accuracy(self, subset_index, true_transmitter):
        '''Test the accuracy of a subset of sensors when detecting the (single) true transmitter
        Args:
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
                mean = self.means[true_x*self.grid_len + true_y, sensor.index]
                std = self.stds[true_x*self.grid_len + true_y, sensor.index]
                data.append(np.random.normal(mean, std))
            for trans in self.transmitters:
                likelihood = trans.multivariant_gaussian.pdf(data)
                self.grid_posterior[trans.x][trans.y] = likelihood * self.grid_priori[trans.x][trans.y] # don't care about
            max_posterior = np.argwhere(self.grid_posterior == np.amax(self.grid_posterior))
            for max_post in max_posterior:  # there might be multiple places with the same highest posterior
                if max_post[0] == true_x and max_post[1] == true_y:
                    count = len(max_posterior)
                    if random.randint(1, count) == 1: # when true transmitter is among the max posterior, randomly pick one
                        success += 1
            i += 1
        return float(success)/test_num


    def select_online_random(self, budget, cores, true_index=-1):
        '''The online random selection
        Args:
            budget (int):
            cores (int):
        '''
        if true_index == -1:
            random.seed()
            true_index = random.randint(0, self.grid_len * self.grid_len)

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

        subset_results = Parallel(n_jobs=cores, verbose=60)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)
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
        Args:
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


    def select_online_nearest(self, budget, cores, true_index=-1):
        '''Online selection using the updated prior information by choosing the 'nearest' sensor
        Args:
            budget (int):
            cores (int):
        '''
        if true_index == -1:
            random.seed()
            true_index = random.randint(0, self.grid_len * self.grid_len)

        self.set_priori()
        plot_data = []
        random.seed(1)
        np.random.seed(2)
        true_transmitter = self.transmitters[true_index]         # in online selection, there is one true transmitter somewhere
        print('true transmitter', true_transmitter)
        self.print_grid(self.grid_priori)

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
            distances = self.weighted_distance_priori(complement_index)
            max_distances = np.argwhere(distances == np.amax(distances))  # there could be multiple max distances
            select = random.choice(max_distances)[0]
            index_nearest = complement_index[select]

            ordered_insert(subset_index, index_nearest)
            complement_index.remove(index_nearest)
            subset_to_compute.append(copy.deepcopy(subset_index))
            self.print_subset(subset_index)
            self.update_hypothesis(true_transmitter, subset_index)
            self.print_grid(self.grid_priori)
            cost += 1

        subset_results = Parallel(n_jobs=cores, verbose=60)(delayed(self.inner_online_accuracy)(true_transmitter, subset_index) for subset_index in subset_to_compute)
        print(subset_results)
        for result in subset_results:
            plot_data.append([str(result[0]), len(result[0]), result[1]])

        return plot_data


    def weighted_distance_priori(self, complement_index):
        '''Compute the weighted distance priori according to the priori distribution for every sensor in
           the complement index list and return the all the distances
        Args:
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
                dist = distance.euclidean([sensor.x, sensor.y], [tran_x, tran_y])
                dist = dist if dist >= 1 else 0.5                                 # sensor very close by with high priori should be selected
                weighted_distance += 1/dist * self.grid_priori[tran_x][tran_y]    # so the metric is priori/disctance

            distances.append(weighted_distance)
        return np.array(distances)


    def select_online_nearest_hetero(self, budget, cores, true_index):
        '''Online selection using the updated prior information by choosing the 'nearest' sensor
        Args:
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
        self.print_grid(self.grid_priori)

        complement_index = [i for i in range(self.sen_num)]
        complement_index.remove(first_index)
        cost = self.sensors[first_index].cost
        subset_to_compute = [copy.deepcopy(subset_index)]
        cost_list = [cost]

        while cost < budget and complement_index:
            print(cost, budget)
            distances = self.weighted_distance_priori(complement_index)
            max_dist_cost = distances[0] / self.sensors[complement_index[0]].cost
            best_candidate = complement_index[0]
            for dist, sen_index in zip(distances, complement_index):
                sen_cost = self.sensors[sen_index].cost
                dist_cost = dist / sen_cost
                if dist_cost > max_dist_cost:
                    max_dist_cost = dist_cost
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
        Args:
            subset_index (np.ndarray, n=1): index of some sensors
        Return:
            (float): o_t_approx
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
        Args:
            subset_index (np.ndarray, n=1): index of some sensors
            cuda_kernal (cuda_kernals.o_t_approx_kernal or o_t_approx_dist_kernal)
        Return:
            (float): o_t_approx
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
        Args:
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


    def scalability_budget(self, budgets):
        '''default # of hypothesis = 32^2 = 1024
           default # of sensors = 100
        Args:
            budgets (list): a list of budgets
        '''
        budget_file = open('plot_data64/budget_', 'w')
        self.grid_len = 32
        self.sen_num = 100
        self.init_transmitters()
        self.set_priori()
        self.init_from_real_data('scalability/gl32_s100/cov', 'scalability/gl32_s100/sensors', 'scalability/gl32_s100/hypothesis')
        for budget in budgets:
            print(budget, end=',', file=budget_file)
            start = time.time()
            self.select_offline_greedy_p_lazy(budget, 12, o_t_approx_kernal)
            print(time.time()-start, end='\n', file=budget_file)


    def scalability_hypothesis(self, grid_lens):
        '''default # of sensors = 100
           default budget = 40
        Args:
            grid_lens (list): a list of grid_lens
        '''
        cov_filename = 'scalability/glCAITAO_s100/cov'
        sensors_filename = 'scalability/glCAITAO_s100/sensors'
        hypothesis_filename = 'scalability/glCAITAO_s100/hypothesis'
        hypothesis_file = open('plot_data64/hypothesis2', 'w')
        self.sen_num = 100
        for grid_len in grid_lens:
            cov = cov_filename.replace('CAITAO', str(grid_len))
            sensors = sensors_filename.replace('CAITAO', str(grid_len))
            hypothesis = hypothesis_filename.replace('CAITAO', str(grid_len))
            self.grid_len = grid_len
            self.init_transmitters()
            self.set_priori()
            self.init_from_real_data(cov, sensors, hypothesis)
            print(grid_len*grid_len, end=',', file=hypothesis_file)
            start = time.time()
            self.select_offline_greedy_p_lazy(40, 6, o_t_approx_kernal)
            print(time.time()-start, end='\n', file=hypothesis_file)


    def scalability_sensor(self, sensor_nums):
        '''default # of hypothesis = 32^2 = 1024
           default budget = 40
        Args:
            sensors (list): a list of # of sensors
        '''
        cov_filename = 'scalability/gl32_sCAITAO/cov'
        sensors_filename = 'scalability/gl32_sCAITAO/sensors'
        hypothesis_filename = 'scalability/gl32_sCAITAO/hypothesis'
        sensor_file = open('plot_data64/sensor', 'w')
        self.grid_len = 32
        for sensor_num in sensor_nums:
            cov = cov_filename.replace('CAITAO', str(sensor_num))
            sensors = sensors_filename.replace('CAITAO', str(sensor_num))
            hypothesis = hypothesis_filename.replace('CAITAO', str(sensor_num))
            self.sen_num = sensor_num
            self.init_transmitters()
            self.set_priori()
            self.init_from_real_data(cov, sensors, hypothesis)
            print(sensor_num, end=',', file=sensor_file)
            start = time.time()
            self.select_offline_greedy_p_lazy(40, 12, o_t_approx_kernal)
            print(time.time()-start, end='\n', file=sensor_file)


    def scalability_budget_online(self, budgets):
        '''default # of hypothesis = 32^2 = 1024
           default # of sensors = 100
        Args:
            budgets (list): a list of budgets
        '''
        budget_file = open('plot_data64/budget_online', 'w')
        self.grid_len = 32
        self.sen_num = 100
        self.init_transmitters()
        self.set_priori()
        self.init_from_real_data('scalability/gl32_s100/cov', 'scalability/gl32_s100/sensors', 'scalability/gl32_s100/hypothesis')
        for budget in budgets:
            print(budget, end=',', file=budget_file)
            start = time.time()
            self.select_online_greedy_p2(budget, 40)
            print(time.time()-start, end='\n', file=budget_file)


    def scalability_hypothesis_online(self, grid_lens):
        '''default # of sensors = 100
           default budget = 20, online need less budget than offline
        Args:
            grid_lens (list): a list of grid_lens
        '''
        cov_filename = 'scalability/glCAITAO_s100/cov'
        sensors_filename = 'scalability/glCAITAO_s100/sensors'
        hypothesis_filename = 'scalability/glCAITAO_s100/hypothesis'
        hypothesis_file = open('plot_data64/hypothesis_online', 'w')
        self.sen_num = 100
        for grid_len in grid_lens:
            cov = cov_filename.replace('CAITAO', str(grid_len))
            sensors = sensors_filename.replace('CAITAO', str(grid_len))
            hypothesis = hypothesis_filename.replace('CAITAO', str(grid_len))
            self.grid_len = grid_len
            self.init_transmitters()
            self.set_priori()
            self.init_from_real_data(cov, sensors, hypothesis)
            print(grid_len*grid_len, end=',', file=hypothesis_file)
            start = time.time()
            self.select_online_greedy_p2(20, 40)
            print(time.time()-start, end='\n', file=hypothesis_file)


    def scalability_sensor_online(self, sensor_nums):
        '''default # of hypothesis = 32^2 = 1024
           default budget = 20
        Args:
            sensors (list): a list of # of sensors
        '''
        cov_filename = 'scalability/gl32_sCAITAO/cov'
        sensors_filename = 'scalability/gl32_sCAITAO/sensors'
        hypothesis_filename = 'scalability/gl32_sCAITAO/hypothesis'
        sensor_file = open('plot_data64/sensor_online', 'w')
        self.grid_len = 32
        for sensor_num in sensor_nums:
            cov = cov_filename.replace('CAITAO', str(sensor_num))
            sensors = sensors_filename.replace('CAITAO', str(sensor_num))
            hypothesis = hypothesis_filename.replace('CAITAO', str(sensor_num))
            self.sen_num = sensor_num
            self.init_transmitters()
            self.set_priori()
            self.init_from_real_data(cov, sensors, hypothesis)
            print(sensor_num, end=',', file=sensor_file)
            start = time.time()
            self.select_online_greedy_p2(20, 40)
            print(time.time()-start, end='\n', file=sensor_file)


def main():
    '''main
    '''
    selectsensor = SelectSensor('config.json')

    #real data
    #selectsensor.init_from_real_data('data32/homogeneous/cov', 'data32/homogeneous/sensors', 'data32/homogeneous/hypothesis')
    selectsensor.scalability_budget_online([1, 5])#, 10, 20, 30, 40, 50, 60, 70, 80])
    selectsensor.scalability_hypothesis_online([16, 24])#, 32, 40, 48, 56, 64, 72, 80])
    selectsensor.scalability_sensor_online([50, 100])#, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

    #selectsensor.init_from_real_data('data64/homogeneous/cov', 'data64/homogeneous/sensors', 'data64/homogeneous/hypothesis')
    #plots.figure_2a(selectsensor)
    #plots.figure_2a(selectsensor)
    #selectsensor.init_from_real_data('data2/homogeneous/cov', 'data2/homogeneous/sensors', 'data2/homogeneous/hypothesis')
    #selectsensor.scalability_budget([90])
    #selectsensor.scalability_hypothesis([16, 24, 32, 40, 48])
    #selectsensor.scalability_hypothesis([80])
    #selectsensor.scalability_sensor([50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000])

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

    #plot_data = selectsensor.select_online_greedy(3, 250)
    #plot_data = selectsensor.select_online_greedy_2(3, 250)
    #start = time.time()
    #plot_data = selectsensor.select_online_greedy_p(3, 4, 250)
    #print('time:', time.time()-start)
    #plots.save_data(plot_data, 'plot_data16/Online_Greedy_v2_.csv')

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
    main()
