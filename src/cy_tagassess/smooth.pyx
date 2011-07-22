# -*- coding: utf8
'''Implementation of the Jelinek-Mercer and Bayes smoothing.'''

from __future__ import division, print_function

cpdef double jelinek_mercer(int local_freq, int sum_locals, int global_freq, 
                            int sum_globals, double lambda_) except *:
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
    
    cdef double local = 0
    if sum_locals >= 0:
        local = (1 - lambda_) * local_freq / sum_locals
    
    cdef double globl = 0
    if sum_globals >= 0:
        globl = lambda_ * global_freq / sum_globals
    
    return local + globl

cpdef double bayes(int local_freq, int sum_locals, int global_freq, 
                   int sum_globals, double lambda_) except *:
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
    
    cdef double globl = 0
    cdef double alpha = 0
    if sum_globals >= 0:
        globl = global_freq / sum_globals

    if local_freq > 0:    
        return (local_freq + lambda_ * globl) /  (sum_locals + lambda_)
    else:
        alpha = lambda_ / (sum_locals + lambda_)
        return alpha * globl