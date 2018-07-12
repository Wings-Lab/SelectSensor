'''
Module for class Transmitter. A transmitter is essentially a hypothesis
'''
class Transmitter:
    '''Encapsulate a transmitter
    Attributes:
        x (int): location - first dimension
        y (int): location - second dimension
        mean_vec (list): mean vector, length is the number of sensors
        multivariant_gaussian(scipy.stats.multivariate_normal):
                              each hypothesis corresponds to a multivariant guassian distribution
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mean_vec = []
        self.multivariant_gaussian = None


    def add_mean_vec(self, mean):
        '''add a mean to mean vector
        '''
        self.mean_vec.append(mean)


    def write_mean_vec(self, filename):
        '''append the mean vector to filename
        '''
        with open(filename, 'a') as f:
            f.write(str(self.mean_vec) + '\n')


    def __str__(self):
        return "(%d, %d) " % (self.x, self.y)


if __name__ == '__main__':
    transmitter = Transmitter(3, 5)
    print(transmitter)
