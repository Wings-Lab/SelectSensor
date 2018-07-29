#!/usr/bin/env python
"""
Script to calculate Mutual Information between two discrete random variables

Roberto maestre - rmaestre@gmail.com
Bojan Mihaljevic - boki.mihaljevic@gmail.com

Caitao Zhan - caitaozhan@stonybrook.edu (minor changes)
"""
from __future__ import division
import math
import time
from numpy  import array, shape, where, in1d
import nose

class InformationTheoryTool:
    '''Compute Entropy and mutual information
    '''
    def __init__(self, data):
        """
        """
        # Check if all rows have the same length
        assert (len(data.shape) == 2)
        # Save data
        self.data = data
        self.n_rows = data.shape[0]
        self.n_cols = data.shape[1]


    def single_entropy(self, x_index, log_base, debug=False):
        """
        Calculate the entropy of a random variable
        """
        # Check if index are into the bounds
        assert (x_index >= 0 and x_index <= self.n_rows)
        # Variable to return entropy
        summation = 0.0
        # Get uniques values of random variables
        values_x = set(self.data[x_index])
        # Print debug info
        if debug:
            print('Entropy of')
            print(self.data[x_index])
        # For each random
        for value_x in values_x:
            px = shape(where(self.data[x_index] == value_x))[1] / self.n_cols
            if px > 0.0:
                summation += px * math.log(px, log_base)
            if debug:
                print('(%d) px:%f' % (value_x, px))
        if summation == 0.0:
            return summation
        else:
            return - summation


    def entropy(self, x_index, y_index, log_base, debug=False):
        """
        Calculate the entropy between two random variable
        """
        assert (x_index >= 0 and x_index <= self.n_rows)
        assert (y_index >= 0 and y_index <= self.n_rows)
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        values_x = set(self.data[x_index])
        values_y = set(self.data[y_index])
        # Print debug info
        if debug:
            print('Entropy between')
            print(self.data[x_index])
            print(self.data[y_index])
        # For each random
        for value_x in values_x:
            for value_y in values_y:
                pxy = len(where(in1d(where(self.data[x_index] == value_x)[0], \
                                where(self.data[y_index] == value_y)[0]) == True)[0]) / self.n_cols
                if pxy > 0.0:
                    summation += pxy * math.log(pxy, log_base)
                if debug:
                    print('(%d,%d) pxy:%f' % (value_x, value_y, pxy))
        if summation == 0.0:
            return summation
        else:
            return - summation



    def mutual_information(self, x_index, y_index, log_base, debug=False):
        """
        Calculate and return Mutual information between two random variables
        """
        # Check if index are into the bounds
        assert (x_index >= 0 and x_index <= self.n_rows)
        assert (y_index >= 0 and y_index <= self.n_rows)
        # Variable to return MI
        summation = 0.0
        # Get uniques values of random variables
        values_x = set(self.data[x_index])
        values_y = set(self.data[y_index])
        # Print debug info
        if debug:
            print('MI between')
            print(self.data[x_index])
            print(self.data[y_index])
        # For each random
        for value_x in values_x:
            for value_y in values_y:
                px = shape(where(self.data[x_index] == value_x))[1] / self.n_cols
                py = shape(where(self.data[y_index] == value_y))[1] / self.n_cols
                pxy = len(where(in1d(where(self.data[x_index] == value_x)[0], \
                                where(self.data[y_index] == value_y)[0]) == True)[0]) / self.n_cols
                if pxy > 0.0:
                    summation += pxy * math.log((pxy / (px*py)), log_base)
                if debug:
                    print('(%d,%d) px:%f py:%f pxy:%f' % (value_x, value_y, px, py, pxy))
        return summation


def test():
    '''some test
    '''
    # Define data array
    data = array([(0, 1, 2, 3, 4, 5, 6, 7, 8, 9),
                  (3, 4, 5, 5, 3, 2, 2, 6, 6, 1),
                  (7, 2, 1, 3, 2, 8, 9, 1, 2, 0),
                  (7, 7, 7, 7, 7, 7, 7, 7, 7, 7),
                  (0, 1, 2, 3, 4, 5, 6, 7, 1, 1)])
    # Create object
    it_tool = InformationTheoryTool(data)

    '''
    # --- Checking single random var entropy

    # entropy of  X_1 (3, 4, 5, 5, 3, 2, 2, 6, 6, 1)
    t_start = time.time()
    print('Entropy(X_0): %f' % it_tool.single_entropy(0, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    # entropy of  X_3 (7, 7, 7, 7, 7, 7, 7, 7, 7, 7)
    t_start = time.time()
    print('Entropy(X_3): %f' % it_tool.single_entropy(3, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    # entropy of  X_4 (0, 1, 2, 3, 4, 5, 6, 7, 8, 9)
    t_start = time.time()
    print('Entropy(X_4): %f' % it_tool.single_entropy(4, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    # --- Checking entropy between two random variables

    # entropy of  X_0 (0, 0, 1, 1, 0, 1, 1, 2, 2, 2) and X_1 (3, 4, 5, 5, 3, 2, 2, 6, 6, 1)
    t_start = time.time()
    print('Entropy(X_0, X_1): %f' % it_tool.entropy(0, 1, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    # entropy of  X_3 (7, 7, 7, 7, 7, 7, 7, 7, 7, 7) and X_3 (7, 7, 7, 7, 7, 7, 7, 7, 7, 7)
    t_start = time.time()
    print('Entropy(X_3, X_3): %f' % it_tool.entropy(3, 3, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))
    '''


    # ---Checking Mutual Information between two random variables

    # Print mutual information between X_0 (0,0,1,1,0,1,1,2,2,2) and X_1 (3,4,5,5,3,2,2,6,6,1)
    t_start = time.time()
    print('MI(X_0, X_1): %f' % it_tool.mutual_information(0, 1, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    # Print mutual information between X_1 (3,4,5,5,3,2,2,6,6,1) and X_2 (7,2,1,3,2,8,9,1,2,0)
    t_start = time.time()
    print('MI(X_1, X_2): %f' % it_tool.mutual_information(1, 2, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    t_start = time.time()
    print('MI(X_0, X_3): %f' % it_tool.mutual_information(0, 3, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    t_start = time.time()
    print('MI(X_0, X_0): %f' % it_tool.mutual_information(0, 0, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    t_start = time.time()
    print('MI(X_0, X_1): %f' % it_tool.mutual_information(0, 1, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    t_start = time.time()
    print('MI(X_0, X_2): %f' % it_tool.mutual_information(0, 2, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))

    t_start = time.time()
    print('MI(X_3, X_3): %f' % it_tool.mutual_information(3, 3, 10))
    print('Elapsed time: %f\n' % (time.time() - t_start))


    # --- Checking results

    # Checking entropy results
    for i in range(0, data.shape[0]):
        assert(it_tool.entropy(i, i, 10) == it_tool.single_entropy(i, 10))

    # Checking mutual information results
    # MI(X,Y) = H(X) + H(Y) - H(X,Y)
    n_rows = data.shape[0]
    i = 0
    while i < n_rows:
        j = i + 1
        while j < n_rows:
            if j != i:
                nose.tools.assert_almost_equal(it_tool.mutual_information(i, j, 10), \
                it_tool.single_entropy(i, 10)+it_tool.single_entropy(j, 10)-it_tool.entropy(i, j, 10))
            j += 1
        i += 1

if __name__ == '__main__':
    test()
