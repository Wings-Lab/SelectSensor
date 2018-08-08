'''
Module for class Sensor
'''
class Sensor:
    '''Encapsulate a sensor
    Attributes:
        x (int):    location - first dimension
        y (int):    location - second dimension
        std(float): each sensor has a standard deviation for receiving signals
        cost (int): each sensor has a engery cost, defualt value is 1
        pre_f (float): value of submodular function in the previous iteration
        gain (float): the gain the context of submodular
        index (int):  a sensor will have an index, like it's ID
    '''
    def __init__(self, x, y, std, cost=1, pre_f=0, gain=0, index=0):
        self.x = x
        self.y = y
        self.std = std
        self.cost = cost
        self.pre_f = pre_f
        self.gain = gain
        self.index = index


    def __str__(self):
        return  "(%d, %d, %f, %d)" % (self.x, self.y, self.gain, self.index)


    def __lt__(self, other):
        '''Override the less than method and turn it into 'more than'
        '''
        return self.gain > other.gain


def main():
    '''Main
    '''
    sensor = Sensor(1, 3, 2.0, gain=1.2, index=0)
    print(sensor)


if __name__ == '__main__':
    main()
