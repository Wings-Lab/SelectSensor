'''
CUDA kernals
'''

import math
from numba import cuda, float64


local_array_size = 24   # Global variables are treated as constants. Always initialize the largest size, i.e. the budget


@cuda.jit('float64(float64)', device=True)
def q_function(x):
    '''q_function(x) = 1 - norf_cdf(x). However, numba does not support scipy.stats.norm.
       So we have to work around this issue. The way to fix this is to use math.erf to substitude scipy.stats.norm
       https://stats.stackexchange.com/questions/187828/how-are-the-error-function-and-standard-normal-distribution-function-related
    '''
    return 1. - 0.5*(1. + math.erf(x/math.sqrt(2.)))


@cuda.jit('void(float64[:,:], int64[:], int64, int64, float64[:])', device=True)
def set_pj_pi(meanvec_array, subset_index, j, i, pj_pi):
    '''1D array minus. C = A - B
    Attributes:
        A, B, C (array-like)
    '''
    index = 0
    for k in subset_index:
        pj_pi[index] = meanvec_array[j, k] - meanvec_array[i, k]
        index += 1


@cuda.jit('float64(float64[:], float64[:,:], float64[:], int64)', device=True)
def matmul(A, B, C, size):
    '''Implement np.dot(np.dot(pj_pi, sub_cov_inv), pj_pi)
       1. C = np.dot(A, B)
       2. return np.dot(C, A)
    Attributes:
        A (1D array): pj_pi
        B (2D array): sub_cov_inv
        C (1D array): tmp array
        size (int64): the size of A, B, C need to be used in computation.
                      note that we allocated a large enough space for every case, in all but one case, some memory is wasted
    Return:
        (float64)
    '''
    for i in range(size):
        summation = 0.
        for k in range(size):
            summation += A[k] * B[k, i]      # C = np.dot(A, B)
        C[i] = summation
    summation = 0.
    for i in range(size):
        summation += C[i] * A[i]             # np.dot(C, A)
    return summation


@cuda.jit('void(float64[:,:], int64[:], float64[:,:], float64, float64[:,:])')
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
    if i < results.shape[0] and j < results.shape[1] and i != j:  # warning: in Linux simulator, need to consider case i ==j
        pj_pi = cuda.local.array(local_array_size, dtype=float64)
        tmp = cuda.local.array(local_array_size, dtype=float64)
        set_pj_pi(meanvec_array, subset_index, j, i, pj_pi)
        results[i, j] = q_function(0.5 * math.sqrt(matmul(pj_pi, sub_cov_inv, tmp, subset_index.size))) * priori


@cuda.jit('void(float64[:,:], int64[:], float64[:,:], float64[:,:])')
def o_t_kernal(meanvec_array, subset_index, sub_cov_inv, results):
    '''The kernal for o_t_approx. Each thread executes a kernal, which is responsible for one element in results array.
    Attributes:
        meanvec_array (np 2D array): contains the mean vector of every transmitter
        subset_index (np 1D array):  index of some sensors
        cov_inv (np 2D array):       inverse of a covariance matrix
        priori (float64):            the prior of each hypothesis
        results (np 2D array):       save the all the results
    '''
    i, j = cuda.grid(2)
    if i < results.shape[0] and j < results.shape[1]:  # warning: in Linux simulator, need to consider case i ==j
        if i == j:
            results[i, j] = 1.
        else:
            pj_pi = cuda.local.array(local_array_size, dtype=float64)
            tmp = cuda.local.array(local_array_size, dtype=float64)
            set_pj_pi(meanvec_array, subset_index, j, i, pj_pi)
            results[i, j] = (1 - q_function(0.5 * math.sqrt(matmul(pj_pi, sub_cov_inv, tmp, subset_index.size))))
