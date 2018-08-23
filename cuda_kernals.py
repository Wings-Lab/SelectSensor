'''
CUDA kernals
'''

import math
import numpy as np
from numba import cuda, float64

local_array_size = 0   # Global variables are treated as constants

@cuda.jit('float64(float64)', device=True)
def q_function(x):
    '''q_function(x) = 1 - norf_cdf(x). However, numba does not support scipy.stats.norm.
       So we have to work around this issue. The way to fix this is to use math.erf to substitude scipy.stats.norm
       https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    '''
    return 1. - 0.5*(1. + math.erf(x/math.sqrt(2.)))


@cuda.jit('void(float64[:], float64[:], float64[:])', device=True)
def array_minus(A, B, C):
    '''1D array minus. C = A - B
    Attributes:
        A, B, C (array-like)
    '''
    for i in range(C.shape[0]):
        C[i] = A[i] - B[i]


@cuda.jit('float64(float64[:], float64[:,:], float64[:])', device=True)
def matmul(A, B, C):
    '''Implement np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)
       1. C = np.dot(A, B)
       2. return np.dot(C, A)
    Attributes:
        A (1D array): pj_pi
        B (2D array): sub_cov_inv
        C (1D array): tmp array
    Return:
        (float64)
    '''
    for i in range(C.shape[0]):
        summation = 0
        for k in range(A.shape[0]):
            summation += A[k] * B[k, i]      # C = np.dot(A, B)
        C[i] = summation
    summation = 0
    for i in range(C.shape[0]):
        summation += C[i] * A[i]             # np.dot(C, A)
    return summation


def update_local_array(size):
    '''update local array size
    '''
    global local_array_size
    local_array_size = size


@cuda.jit('void(float64[:,:], float64[:], float64[:,:], float64, float64[:,:])')
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
        pj_pi = cuda.local.array(local_array_size, dtype=float64)
        tmp = cuda.local.array(local_array_size, dtype=float64)
        array_minus(meanvec_array[j][subset_index], meanvec_array[i][subset_index], pj_pi)
        results[i, j] = q_function(0.5 * math.sqrt(matmul(pj_pi, sub_cov_inv, tmp))) * priori
        #results[i, j] = q_function(0.5 * math.sqrt(np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi))) * priori
        #print((i, j, results[i, j]))
