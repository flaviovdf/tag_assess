# -*- coding: utf8
#cython: cdivision = True

'''Implementation of the Jelinek-Mercer and Bayes smoothing.'''

from __future__ import division, print_function

cdef double NAN = float('nan')

cpdef double jelinek_mercer(int local_freq, int sum_locals, int global_freq, 
                            int sum_globals, double lambda_):
    '''
    Calculates the jelinek mercer smoothing for the given parameters.
    
    Arguments
    ---------
    local_freq: int
        The local frequency of the tag. By local we mean, the amount of times
        it was used by a given user or it annotated an item.
    sum_locals: int
        Equivalent to the sum of local frequencies for all tags in that item
        or user
    global_freq: int
        How many times the tag appears in the collections
    sum_globals: int
        For all tags, the sum of many times those tags appear in the 
        collection
    lambda_: double
        the hyper parameter
    '''
    if sum_locals == 0 or sum_globals == 0:
        return NAN
    
    cdef double local = <double>((1 - lambda_) * local_freq) / sum_locals
    cdef double globl = <double>(lambda_ * global_freq) / sum_globals
    return local + globl

cpdef double bayes(int local_freq, int sum_locals, int global_freq, 
                   int sum_globals, double lambda_):
    '''
    Calculates the bayes smoothing for the given parameters.
    
    Arguments
    ---------
    local_freq: int
        The local frequency of the tag. By local we mean, the amount of times
        it was used by a given user or it annotated an item.
    sum_locals: int
        Equivalent to the sum of local frequencies for all tags in that item
        or user
    global_freq: int
        How many times the tag appears in the collections
    sum_globals: int
        For all tags, the sum of many times those tags appear in the 
        collection
    lambda_: double
        the hyper parameter
    '''
    if (sum_locals + lambda_) == 0 or sum_globals == 0:
        return NAN

    cdef double globl = <double>(global_freq) / sum_globals
    cdef double alpha = 0
    
    if local_freq > 0:    
        return <double>(local_freq + lambda_ * globl) /  (sum_locals + lambda_)
    else:
        alpha = lambda_ / (sum_locals + lambda_)
        return alpha * globl
