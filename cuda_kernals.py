'''
CUDA kernals
'''

import math
from scipy.stats import norm
import numpy as np
from numba import cuda

@cuda.jit
def o_t_approx_kernal(meanvec_array, subset_index, sub_cov_inv, results):
    '''the kernal for o_t_approx
    Attributes:
        meanvec_array (np 2D array): contains the mean vector of every transmitter
        subset_index (np 1D array): index of some sensors
        cov_inv (np 2D array): inverse of a covariance matrix
        results (np 2D array): save the all the results
    '''
    i, j = cuda.grid(2)
    pj_pi = meanvec_array[i][subset_index] - meanvec_array[j][subset_index]
    results[i, j] = norm.sf(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)))
