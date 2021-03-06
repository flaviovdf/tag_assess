# -*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''Module which contains functions to calculate entropy related metrics'''
from __future__ import division, print_function

from math import log

import numpy as np
cimport numpy as np
np.import_array()

INF = np.inf

#Log2 from C99
cdef extern from "math.h":
    double log2(double)

cpdef double entropy(np.ndarray[np.float_t, ndim=1] probabilities_x):
    '''
    Calculates the entropy (H) of the input vector which
    represents some random variable X.

    Arguments
    ---------
    probabilities_x: numpy array or any iterable
        Array with the individual probabilities_x. Values must be 0 <= x <=1
    '''
    cdef np.float_t return_val = 0
    cdef Py_ssize_t i = 0
    
    for i in range(probabilities_x.shape[0]):
        if probabilities_x[i] > 0:
            return_val -= probabilities_x[i] * log2(probabilities_x[i])
    
    return return_val

cpdef double mutual_information(
        np.ndarray[np.float_t, ndim=1] probabilities_x, 
        np.ndarray[np.float_t, ndim=1] probabilities_xy):
    '''
    Calculates the mutual information between the
    random variables (X and Y):

    Arguments
    ---------
    probabilities_x: numpy array or any iterable
        Array with the individual probabilities X. Values must be 0 <= x <= 1

    probabilities_xy: numpy array or any iterable
        Array with the individual probabilities for X|Y. Values must be 
        0 <= x <= 1
    '''

    cdef np.float_t h_x = entropy(probabilities_x)
    cdef np.float_t h_xy = entropy(probabilities_xy)
    return h_x - h_xy

cpdef double norm_mutual_information(
        np.ndarray[np.float_t, ndim=1] probabilities_x, 
        np.ndarray[np.float_t, ndim=1] probabilities_xy):
    '''
    Calculates the normalized mutual information between the
    random variables (X and X|Y):

    Arguments
    ---------
    probabilities_x: numpy array or any iterable
        Array with the individual probabilities X. Values must be 0 <= x <= 1

    probabilities_xy: numpy array or any iterable
        Array with the individual probabilities for X|Y. 
        Values must be 0 <= x <= 1
    '''

    cdef np.float_t h_x = entropy(probabilities_x)
    cdef np.float_t h_xy = entropy(probabilities_xy)

    cdef np.float_t normalized_mi = 0
    if h_x > 0 and h_xy > 0:
        normalized_mi = 1 - (h_x - h_xy) / h_x
        
    return normalized_mi

cpdef double kullback_leiber_divergence(
        np.ndarray[np.float_t, ndim=1] probabilities_p, 
        np.ndarray[np.float_t, ndim=1] probabilities_q):
    '''
    Calculates the Kullback-Leiber divergence between the distributions
    of two random variables.

    $$ D_{kl}(P(X) || Q(X)) = \sum_{x \in X) p(x) * log(\frac{p(x)}{q(x)}) $$

    Arguments
    ---------
    probabilities_p: numpy array or any iterable
        Array with the individual probabilities P. Values must be 0 <= x <= 1

    probabilities_q: numpy array or any iterable
        Array with the individual probabilities for Q. Values must be 0 <= x <= 1
    '''
    assert probabilities_p.shape[0] == probabilities_q.shape[0]

    cdef np.float_t return_val = 0
    cdef Py_ssize_t i = 0
    cdef np.float_t prob_p = 0
    cdef np.float_t prob_q = 0

    for i in range(probabilities_p.shape[0]):
        prob_p = probabilities_p[i] 
        prob_q = probabilities_q[i]

        if prob_p != 0 and prob_q == 0:
            return INF
        elif prob_p > 0 and prob_q > 0:
            return_val += prob_p * (log2(prob_p) - log2(prob_q))
    return return_val