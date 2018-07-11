'''
Select sensor and detect transmitter
'''
import random
import traceback
import numpy as np
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
        transmitters (list): a 2D list of Transmitter
        sensors (dict):      a dictionary of Sensor. less than 10% the # of transmitters
        grid (np.ndarray):   the element is the probability of hypothesis - transmitter
        priori_mean (list):  list of mean of the priori 2D normal distribution
        priori_cov  (list):  list of covariance of the priori 2D normal distribution
        data (list):         a 2D list of observation data
    '''
    def __init__(self, filename):
        self.config = read_config(filename)
        self.sen_num = int(self.config["sensor_number"])
        self.grid_len = int(self.config["grid_length"])
        self.transmitters = []
        self.sensors = {}
        self.init_random_sensors()
        self.init_transmitters()
        self.set_priori()
        self.data = []


    def set_priori(self):
        '''Set priori distribution - 2D normal distribution
           The mean and covariance come from configuration file
        '''
        x, y = np.mgrid[0:50:1, 0:50:1]
        pos = np.zeros(x.shape + (2,))
        pos[:, :, 0] = x
        pos[:, :, 1] = y

        try:
            mean = self.config['priori_mean']
            mean = mean.split()
            cov = self.config['priori_cov']
            cov = cov.split()
            self.priori_mean = [float(mean[0]), float(mean[1])]
            self.priori_cov = [[float(cov[0]), float(cov[1])], [float(cov[2]), float(cov[3])]]
            self.grid = multivariate_normal(self.priori_mean, self.priori_cov).pdf(pos)
        except IndexError:
            traceback.print_exc()
        except Exception:
            traceback.print_exc()


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
                #dist = distance.euclidean([x, y], [self.transmitter.x, self.transmitter.y])
                self.sensors[(x, y)] = Sensor(x, y)
                i += 1


    def data_imputation(self):
        '''Since we don't have the real data yet, we make up some fake data
        '''
        means_stds = {}   # assume data of one pair of transmitter-sensor is normal distributed
        for transmitter in self.transmitters:
            tran_x, tran_y = transmitter.x, transmitter.y
            for sensor in self.sensors:
                sen_x, sen_y = sensor.x, sensor.y
                dist = distance.euclidean([sen_x, sen_y], [tran_x, tran_y])
                dist = 1e-3 if dist < 1e-3 else dist  # in case distance is zero
                mean = 1/dist
                std = random.uniform(3, 5)
                means_stds[(tran_x, tran_y, sen_x, sen_y)] = (mean, std)
        i = 0
        while i < 1000: # sample 1000 times, each time 2500 rows with 200 columns
            for transmitter in self.transmitters:
                tran_x, tran_y = transmitter.x, transmitter.y
                hypothesis = []
                for sensor in self.sensors:
                    sen_x, sen_y = sensor.x, sensor.y
                    mean, std = means_stds.get((tran_x, tran_y, sen_x, sen_y))
                    hypothesis.append(np.random.normal(mean, std))
            i += 1




    def compute_posterior(self):
        '''Given priori probability and observation data, compute posterior probability
        '''


    def print(self):
        '''Print for testing
        '''
        print('sensors:\n')
        for key in self.sensors:
            print(self.sensors[key])
        print('\nhypothesis:\n')
        print(self.grid.shape, '\n', self.grid)


def main():
    '''main
    '''
    select_sensor = SelectSensor('config.json')
    select_sensor.print()


if __name__ == '__main__':
    main()
