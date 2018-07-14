'''
Module for class Sensor
'''
class Sensor:
    '''Encapsulate a sensor
    Attributes:
        x (int):    location - first dimension
        y (int):    location - second dimension
        std(float): each sensor has a standard deviation for receiving signals
    '''
    def __init__(self, x, y, std):
        self.x = x
        self.y = y
        self.std = std


    def __str__(self):
        return  "(%d, %d)" % (self.x, self.y)


    def output(self):
        '''Output into files
        '''
        return "%d %d %f\n" % (self.x, self.y, self.std)


def main():
    '''Main
    '''
    sensor = Sensor(1, 3, 2.0)
    s = sensor.output()
    print(s)


if __name__ == '__main__':
    main()
