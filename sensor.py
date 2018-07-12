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


    def output(self):
        '''Output into files
        '''
        return "%d %d\n" % (self.x, self.y)


def main():
    '''Main
    '''
    sensor = Sensor(1, 3)
    s = sensor.output()
    print(s)


if __name__ == '__main__':
    main()
