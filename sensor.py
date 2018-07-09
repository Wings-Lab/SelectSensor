import numpy as np

class Sensor:
    '''Encapsulate a sensor
    
    Attributes:
        x (int): location - first dimension
        y (int): location - second dimension
        mean (float): observation - output of the sensors
        std  (float): observation - output of the sensors
    '''
    def __init__(self, x, y, mean, std):
        self.x = x
        self.y = y
        self.mean = mean
        self.std  = std


    def __str__(self):
        str1 = "(%d, %d)" % (self.x, self.y)
        str2 = "mean = %f  std = %f".ljust(10) % (self.mean, self.std)
        return  str1.ljust(10) + str2


if __name__ == '__main__':
    sensor = Sensor(1, 3, 5, 7)
    print(sensor)