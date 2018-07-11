'''
Module for class Transmitter
'''
class Transmitter:
    '''Encapsulate a transmitter
    Attributes:
        x (int): location - first dimension
        y (int): location - second dimension
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y


    def __str__(self):
        return "(%d, %d) " % (self.x, self.y)


if __name__ == '__main__':
    transmitter = Transmitter(3, 5)
    print(transmitter)
