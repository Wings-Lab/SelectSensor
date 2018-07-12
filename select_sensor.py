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
        grid (np.ndarray):   the element is the probability of hypothesis - transmitter
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
        self.grid = np.zeros(0)
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


    def read_init_sensor(self, filename):
        '''Read location of sensors and init the sensors
        '''
        self.sensors = {}
        with open(filename, 'r') as f:
            lines = f.readlines()
            for line in lines:
                line = line.split(' ')
                x, y = int(line[0]), int(line[1])
                self.sensors[(x, y)] = Sensor(x, y)


    def mean_std_output(self):
        '''test mean std input output
        '''
        with open('mean_std.txt', 'w') as f:
            for transmitter_list in self.transmitters:
                for transmitter in transmitter_list:
                    tran_x, tran_y = transmitter.x, transmitter.y
                    for sensor in self.sensors:
                        sen_x, sen_y = sensor[0], sensor[1]   # key -> value, key is enough to get (x,y)
                        dist = distance.euclidean([sen_x, sen_y], [tran_x, tran_y])
                        dist = 1e-2 if dist < 1e-2 else dist  # in case distance is zero
                        mean = 800/dist + 2
                        std = random.uniform(1, 2)
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
        '''Since we don't have the real data yet, we make up some artificial data
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
                        one_transmitter.append(round(np.random.normal(mean, std), 3))   # 0.123
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

    
    #select_sensor.read_init_sensor('sensor.txt')
    #select_sensor.mean_std_output()
    
    
    select_sensor.read_init_sensor('sensor.txt')
    select_sensor.read_mean_std('mean_std.txt')
    select_sensor.generate_data()
    

    #select_sensor.compute_multivariant_gaussian()
    #select_sensor.init_random_sensors()
    #select_sensor.save_sensor()
    #select_sensor.print()


if __name__ == '__main__':
    main()
