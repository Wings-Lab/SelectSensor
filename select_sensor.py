'''
Select sensor and detect transmitter
'''
import random
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
        grid (np.ndarray):   the element is the probability of data - transmitter
        transmitters (list): a 2D list of Transmitter
        sensors (dict):      a dictionary of Sensor. less than 10% the # of transmitters
        data (ndarray):      a 2D array of observation data
        covariance (list):   a 2D list of covariance. each data share a same covariance matrix
    '''
    def __init__(self, filename):
        self.config = read_config(filename)
        self.sen_num = int(self.config["sensor_number"])
        self.grid_len = int(self.config["grid_length"])
        self.grid = np.zeros(0)
        self.transmitters = []
        self.sensors = {}
        self.data = np.zeros(0)
        self.covariance = []
        self.init_transmitters()
        self.set_priori()


    def set_priori(self):
        '''Set priori distribution - uniform distribution
        '''
        uniform = 1./(self.grid_len * self.grid_len)
        self.grid = np.full((self.grid_len, self.grid_len), uniform)


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
            if self.sensors.get((x, y)):
                continue
            else:    # no sensor exists at (x,y)
                self.sensors[(x, y)] = Sensor(x, y)
                i += 1


    def save_sensor(self):
        '''Save location of sensors
        '''
        with open('sensor.txt', 'w') as f:
            for key in self.sensors:
                f.write(self.sensors[key].output())


    def read_init_sensor(self):
        '''Read location of sensors and init the sensors
        '''
        self.sensors = {}
        with open('sensor.txt', 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y = int(line[0]), int(line[1])
                self.sensors[(x, y)] = Sensor(x, y)


    def data_imputation(self):
        '''Since we don't have the real data yet, we make up some fake data
           Then save them in a csv file
        '''
        means_stds = {}   # assume data of one pair of transmitter-sensor is normal distributed
        for transmitter_list in self.transmitters:
            for transmitter in transmitter_list:
                tran_x, tran_y = transmitter.x, transmitter.y
                for sensor in self.sensors:
                    sen_x, sen_y = sensor[0], sensor[1]   # key -> value, key is enough to get (x,y)
                    dist = distance.euclidean([sen_x, sen_y], [tran_x, tran_y])
                    dist = 1e-2 if dist < 1e-2 else dist  # in case distance is zero
                    mean = 300/dist + 5
                    std = random.uniform(3, 5)
                    means_stds[(tran_x, tran_y, sen_x, sen_y)] = (mean, std)
        data = []
        i = 0
        while i < 100: # sample 1000 times
            for transmitter_list in self.transmitters:
                for transmitter in transmitter_list:
                    tran_x, tran_y = transmitter.x, transmitter.y
                    data = []
                    for sensor in self.sensors:
                        sen_x, sen_y = sensor[0], sensor[1]
                        mean, std = means_stds.get((tran_x, tran_y, sen_x, sen_y))
                        data.append(round(np.random.normal(mean, std), 3))   # 0.123
                    data.append(data)
            if int(i%5) == 0:
                print(i)
            i += 1
        data_pd = pd.DataFrame(data)
        data_pd.to_csv('./artificial_samples.csv', index=False, header=False)


    def compute_multivariant_gaussian(self):
        '''Given observation data, compute the guassian function.
           Each hypothesis corresponds to a single gaussian function
           with different mean but the same covariance.
        '''
        data = pd.read_csv('./artificial_samples.csv', header=None)
        self.covariance = np.cov(data.as_matrix().T)
        print('Computed covariance!')
        for transmitter_list in self.transmitters:
            for transmitter in transmitter_list:
                tran_x, tran_y = transmitter.x, transmitter.y
                mean_vector = []
                for sensor in self.sensors:
                    sen_x, sen_y = sensor[0], sensor[1]   # key -> value, key is enough to get (x,y)
                    dist = distance.euclidean([sen_x, sen_y], [tran_x, tran_y])
                    dist = 1e-2 if dist < 1e-2 else dist  # in case distance is zero
                    mean = 300/dist + 5
                    mean_vector.append(round(mean, 3))    # generate mean vector
                transmitter.multivariant_gaussian = multivariate_normal(mean=mean_vector, cov=self.covariance)


    def print(self):
        '''Print for testing
        '''
        print('\ndata:\n')
        print(self.grid.shape, '\n', self.grid)
        print(self.covariance)


def main():
    '''main
    '''
    select_sensor = SelectSensor('config.json')
    select_sensor.read_init_sensor()
    select_sensor.compute_multivariant_gaussian()
    #select_sensor.init_random_sensors()
    #select_sensor.save_sensor()
    #select_sensor.data_imputation()
    #select_sensor.print()


if __name__ == '__main__':
    main()
