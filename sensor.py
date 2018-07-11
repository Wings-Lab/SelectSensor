'''
Module for class Sensor
'''
class Sensor:
    '''Encapsulate a sensor
    Attributes:
        x (int): location - first dimension
        y (int): location - second dimension
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __str__(self):
        return  "(%d, %d)" % (self.x, self.y)


if __name__ == '__main__':
    sensor = Sensor(1, 3)
    print(sensor)
