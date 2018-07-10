import numpy as np
import random
from scipy.spatial import distance
from scipy.stats import multivariate_normal
from sensor import Sensor
from transmitter import Transmitter
from utility import read_config


class SelectSensor:
    '''Near-optimal low-cost sensor selection
    
    Attributes:
        sensors (dict): a dictionary of sensors
        sen_num (int): the number of sensors
        grid (np.ndarray): the element is the probability of hypothesis - transmitter
        grid_len (int): the length of the grid
        transmitter (Transmitter): transmitter - intruder
        priori_mean (list): list of mean of the priori 2D normal distribution
        priori_cov  (list): list of covariance of the priori 2D normal distribution
    '''
    def __init__(self, filename):
        self.config = read_config(filename)
        self.grid_len = int(self.config["grid_length"])
        self.sen_num = int(self.config["sensor_number"])
        self.set_priori()
        self.sensors = {}
        self.transmitter = Transmitter(random.randint(0, self.grid_len-1), random.randint(0, self.grid_len-1))
        i = 0
        while i < self.sen_num:
            x = random.randint(0, self.grid_len-1) # randomly find a place for a sensor
            y = random.randint(0, self.grid_len-1)
            if self.sensors.get((x,y)):  
                continue
            else:    # no sensor exists at (x,y)
                dist = distance.euclidean([x, y], [self.transmitter.x, self.transmitter.y])
                self.sensors[(x,y)] = Sensor(x, y, 1/dist, random.uniform(3, 5))
                i += 1


    def set_priori(self):
        '''Set priori distribution - 2D normal distribution
           The mean and covariance come from configuration file
        '''
        x, y = np.mgrid[0:50:1, 0:50:1]
        pos = np.zeros(x.shape + (2,))
        pos[:, :, 0] = x; pos[:, :, 1] = y

        try:
            mean = self.config['priori_mean']
            mean = mean.split()
            self.priori_mean = [int(mean[0]), int(mean[1])]
            cov = self.config['priori_cov']
            cov = cov.split()
            self.priori_cov = [[int(cov[0]), int(cov[1])], [int(cov[2]), int(cov[3])]]
            self.grid = multivariate_normal(self.priori_mean, self.priori_cov).pdf(pos)
        except Exception as e:
            print(e)
        


    def print(self):
        print('transmitter: ', self.transmitter, '\n')
        print('sensors:\n')
        for key in self.sensors:
            print(self.sensors[key])
        print('\nhypothesis:\n')
        print(self.grid.shape, '\n', self.grid)


if __name__ == '__main__':
    selectSensor = SelectSensor('config.json')
    selectSensor.print()

    

