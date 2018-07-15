'''
Module for class Transmitter. A transmitter is essentially a hypothesis
'''
class Transmitter:
    '''Encapsulate a transmitter
    Attributes:
        x (int): location -  first dimension
        y (int): location -  second dimension
        mean_vec (list):     mean vector, length is the number of sensors
        mean_vec_sub (list): mean vector for subset of sensors
        multivariant_gaussian(scipy.stats.multivariate_normal):
                              each hypothesis corresponds to a multivariant guassian distribution
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.mean_vec = []
        self.mean_vec_sub = []
        self.multivariant_gaussian = None
        self.error = 0


    def write_mean_vec(self, filename):
        '''append the mean vector to filename
        '''
        with open(filename, 'a') as f:      # when the file already exists, it will not overwirte
            f.write(str(self.mean_vec) + '\n')


    def add_error(self):
        '''Error counter
        '''
        self.error += 1


    def __str__(self):
        str1 = "(%d, %d) ".ljust(10) % (self.x, self.y)
        return str1 + str(self.error)


if __name__ == '__main__':
    transmitter = Transmitter(3, 5)
    transmitter2 = Transmitter(7, 9)
    print(transmitter)
    print(transmitter2)
