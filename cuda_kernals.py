'''
CUDA kernals
'''

import math
from scipy.stats import norm
import numpy as np
from numba import cuda


@cuda.jit
def o_t_approx_kernal(meanvec_array, subset_index, sub_cov_inv, priori, results):
    '''The kernal for o_t_approx. Each thread executes a kernal, which is responsible for one element in results array.
    Attributes:
        meanvec_array (np 2D array): contains the mean vector of every transmitter
        subset_index (np 1D array):  index of some sensors
        cov_inv (np 2D array):       inverse of a covariance matrix
        priori (float64):            the prior of each hypothesis
        results (np 2D array):       save the all the results
    '''
    i, j = cuda.grid(2)
    if i < results.shape[0] and j < results.shape[1] and i != j:
        pj_pi = np.array(meanvec_array[j][subset_index]) - np.array(meanvec_array[i][subset_index])
        results[i, j] = norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))) * priori
        #print((i, j, results[i, j]))
