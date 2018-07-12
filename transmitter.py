'''
Module for class Transmitter. A transmitter is essentially a hypothesis
'''
class Transmitter:
    '''Encapsulate a transmitter
    Attributes:
        x (int): location - first dimension
        y (int): location - second dimension
        multivariant_gaussian(scipy.stats.multivariate_normal):
                              each hypothesis corresponds to a multivariant guassian distribution
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.multivariant_gaussian = None


    def __str__(self):
        return "(%d, %d) " % (self.x, self.y)


if __name__ == '__main__':
    transmitter = Transmitter(3, 5)
    print(transmitter)
