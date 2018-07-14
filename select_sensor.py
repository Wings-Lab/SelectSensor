'''
Select sensor and detect transmitter
'''
import random
import math
#import traceback
import numpy as np
import pandas as pd
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sensor import Sensor
from transmitter import Transmitter
from utility import read_config



class SelectSensor:
    '''Near-optimal low-cost sensor selection

    Attributes:
        config (json):       configurations - settings and parameters
        sen_num (int):       the number of sensors
        grid_len (int):      the length of the grid
        grid_priori (np.ndarray):    the element is priori probability of hypothesis - transmitter
        grid_posterior (np.ndarray): the element is posterior probability of hypothesis - transmitter
        transmitters (list): a 2D list of Transmitter
        sensors (dict):      a dictionary of Sensor. less than 10% the # of transmitters
        data (ndarray):      a 2D array of observation data
        covariance (list):   a 2D list of covariance. each data share a same covariance matrix
        mean_stds (dict):    assume sigal between a transmitter-sensor pair is normal distributed
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


    def set_priori(self):
        '''Set priori distribution - uniform distribution
        '''
        uniform = 1./(self.grid_len * self.grid_len)
        self.grid_priori = np.full((self.grid_len, self.grid_len), uniform)


    def init_transmitters(self):
        '''Initiate a transmitter at all locations
        '''
        for i in range(self.grid_len):
            temp_transmitters = []
            for j in range(self.grid_len):
                temp_transmitters.append(Transmitter(i, j))
            self.transmitters.append(temp_transmitters)


    def init_random_sensors(self):
        '''Initiate some sensors randomly
        '''
        i = 0
        while i < self.sen_num:
            x = random.randint(0, self.grid_len-1) # randomly find a place for a sensor
            y = random.randint(0, self.grid_len-1)
            if self.sensors.get((x, y)): # a sensor exists at (x, y)
                continue
            else:                        # no sensor exists at (x,y)
                self.sensors[(x, y)] = Sensor(x, y, random.uniform(1, 2))
                i += 1


    def save_sensor(self):
        '''Save location of sensors
        '''
        with open('sensor.txt', 'w') as f:
            for key in self.sensors:
                f.write(self.sensors[key].output())


    def read_init_sensor(self, filename):
        '''Read location of sensors and init the sensors
        '''
        self.sensors = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y, std = int(line[0]), int(line[1]), float(line[2])
                self.sensors[(x, y)] = Sensor(x, y, std)


    def mean_std_output(self):
        '''test mean std input output
        '''
        with open('mean_std.txt', 'w') as f:
            for transmitter_list in self.transmitters:
                for transmitter in transmitter_list:
                    tran_x, tran_y = transmitter.x, transmitter.y
                    for key in self.sensors:
                        sen_x, sen_y, std = self.sensors[key].x, self.sensors[key].y, self.sensors[key].std
                        dist = distance.euclidean([sen_x, sen_y], [tran_x, tran_y])
                        dist = 0.5 if dist < 1e-2 else dist  # in case distance is zero
                        mean = 100 - 20*math.log(2*dist)
                        f.write("%d %d %d %d %f %f\n" % (tran_x, tran_y, sen_x, sen_y, mean, std))


    def read_mean_std(self, filename):
        '''read mean std information between transmitters and sensors
        '''
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                tran_x, tran_y = int(line[0]), int(line[1])
                sen_x, sen_y = int(line[2]), int(line[3])
                mean, std = float(line[4]), float(line[5])
                self.means_stds[(tran_x, tran_y, sen_x, sen_y)] = (mean, std)


    def generate_data(self):
        '''Since we don't have the real data yet, we make up some artificial data according to mean_std.txt
           Then save them in a csv file. also save the mean vector
        '''
        total_data = []
        for transmitter_list in self.transmitters:
            for transmitter in transmitter_list:
                tran_x, tran_y = transmitter.x, transmitter.y
                data = []
                i = 0
                while i < 100:                   # sample 100 times for each transmitter
                    one_transmitter = []
                    for sensor in self.sensors:  # for each transmitter, send signal to all sensors
                        sen_x, sen_y = sensor[0], sensor[1]
                        mean, std = self.means_stds.get((tran_x, tran_y, sen_x, sen_y))
                        one_transmitter.append(round(np.random.normal(mean, std), 4))   # 0.1234
                    data.append(one_transmitter)
                    total_data.append(one_transmitter)
                    i += 1
                data = np.array(data)
                mean_vec = data.mean(axis=0).tolist()
                setattr(transmitter, 'mean_vec', mean_vec)
                transmitter.write_mean_vec('mean_vector.txt')

        data_pd = pd.DataFrame(total_data)
        data_pd.to_csv('artificial_samples.csv', index=False, header=False)


    def compute_multivariant_gaussian(self):
        '''Read data and mean vectors, then compute the guassian function by using the data
           Each hypothesis corresponds to a single gaussian function
           with different mean but the same covariance.
        '''
        data = pd.read_csv('artificial_samples.csv', header=None)
        self.covariance = np.cov(data.as_matrix().T)
        print('Computed covariance!')
        with open('mean_vector.txt', 'r') as f:
            for transmitter_list in self.transmitters:
                for transmitter in transmitter_list:
                    line = f.readline()
                    line = line[1:-2]
                    line = line.split(', ')
                    mean_vector = [float(i) for i in line]
                    setattr(transmitter, 'mean_vec', mean_vector)
                    setattr(transmitter, 'multivariant_gaussian', multivariate_normal(mean=mean_vector, cov=self.covariance))


    def test_error(self):
        '''Generate new data, calculate posterior probability, compute classification error.
           For each transmitter, test 10 times
        '''
        total_test = 0
        error = 0
        self.grid_posterior = np.zeros((self.grid_len, self.grid_len))
        for transmitter_list in self.transmitters:
            for transmitter in transmitter_list:
                tran_x, tran_y = transmitter.x, transmitter.y
                if tran_x == tran_y:
                    print(tran_x)
                i = 0
                while i < 10:  # test 10 times for each transmitter
                    data = []
                    for sensor in self.sensors:
                        sen_x, sen_y = sensor[0], sensor[1]
                        mean, std = self.means_stds.get((tran_x, tran_y, sen_x, sen_y))
                        data.append(round(np.random.normal(mean, std), 3))
                    for transmitter_list2 in self.transmitters:
                        for transmitter2 in transmitter_list2:  # given hypothesis, the probability of data
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

    def print(self):
        '''Print for testing
        '''
        for transmitter_list in self.transmitters:
            for transmitter in transmitter_list:
                print(transmitter)
        #print('\ndata:\n')
        #print(self.grid.shape, '\n', self.grid)
        #print(self.covariance)


def main():
    '''main
    '''

    select_sensor = SelectSensor('config.json')

    select_sensor.read_init_sensor('sensor.txt')
    select_sensor.read_mean_std('mean_std.txt')
    select_sensor.compute_multivariant_gaussian()
    print('error ', select_sensor.test_error())
    select_sensor.print()

    #select_sensor.read_init_sensor('sensor.txt')
    #select_sensor.mean_std_output()

    #select_sensor.read_init_sensor('sensor.txt')
    #select_sensor.read_mean_std('mean_std.txt')
    #select_sensor.generate_data()

    #select_sensor.init_random_sensors()
    #select_sensor.save_sensor()
    #select_sensor.print()


if __name__ == '__main__':
    main()
