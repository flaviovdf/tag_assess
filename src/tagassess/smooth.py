# -*- coding: utf8
'''Implementation of the Jelinek-Mercer and Bayes smoothing.'''

from __future__ import division, print_function

def jelinek_mercer(local_freq, sum_locals, global_freq, sum_globals, lambda_):
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
    local = 0
    if sum_locals:
        local = (1 - lambda_) * local_freq / sum_locals
    
    globl = 0
    if sum_globals:
        globl = lambda_ * global_freq / sum_globals
    
    prob = local + globl
    return prob

def bayes(local_freq, sum_locals, global_freq, sum_globals, lambda_):
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
    
    globl = 0
    if sum_globals:
        globl = global_freq / sum_globals
    
    if local_freq > 0:
        prob = (local_freq + lambda_ * globl) /  (sum_locals + lambda_)
    else:
        alpha = lambda_ / (sum_locals + lambda_)
        prob = alpha * globl
    return prob