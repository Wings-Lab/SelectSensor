import numpy as np
import random
from scipy.spatial import distance
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
    '''
    def __init__(self, filename):
        config = read_config(filename)
        self.grid_len = int(config["grid_length"])
        self.sen_num = int(config["sensor_number"])
        self.grid = np.zeros((self.grid_len, self.grid_len))
        self.sensors = {}
        self.transmitter = Transmitter(random.randint(0, self.grid_len-1), random.randint(0, self.grid_len-1))
        i = 0
        while i < self.sen_num:
            x = random.randint(0, self.grid_len-1)
            y = random.randint(0, self.grid_len-1)
            if self.sensors.get((x,y)):  # no sensor exist at (x,y)
                continue
            else:    
                dist = distance.euclidean([x, y], [self.transmitter.x, self.transmitter.y])
                self.sensors[(x,y)] = Sensor(x, y, 1/dist, random.uniform(3, 5))
                i += 1


    def print(self):
        print('transmitter: ', self.transmitter, '\n')
        print('sensors:\n')
        for key in self.sensors:
            print(self.sensors[key])


if __name__ == '__main__':
    selectSensor = SelectSensor('config.json')
    selectSensor.print()

