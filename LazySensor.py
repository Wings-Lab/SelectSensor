'''Custmoized object for sorting
'''

class LazySensor:
    '''In lazy greedy algorithm, the gain of submodular function will be sorted from high to low.
    Attributes:
        gain (float): the gain of submodular function
        index (int): the index of a sensor
    '''
    def __init__(self, gain, index):
        self.gain = gain
        self.index = index

    def __lt__(self, other):          # redefine less than, to make it 'more than'
        return self.gain > other.gain

    def __str__(self):
        return str(self.gain) + ' ' + str(self.index)


if __name__ == '__main__':
    ls0 = LazySensor(3.1, 0)
    ls1 = LazySensor(2.9, 1)
    ls2 = LazySensor(3.2, 2)
    ls3 = LazySensor(3.0, 3)
    ls4 = LazySensor(4.1, 4)
    ls5 = LazySensor(2.0, 5)
    ls6 = LazySensor(5.2, 6)
    ls7 = LazySensor(1.0, 7)
    mylist = [ls0, ls1, ls2, ls3, ls4, ls5, ls6, ls7]
    for e in mylist:
        print(e)
    print('*****')
    mylist.sort()
    for e in mylist:
        print(e)
    