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
    
    Returns
    -------
    A tuple with the smoothed probability and the alpha estimate.
    '''
    local = 0
    if sum_locals:
        local = (1 - lambda_) * local_freq / sum_locals
    
    globl = 0
    if sum_globals:
        globl = lambda_ * global_freq / sum_globals
    
    prob = local + globl
    return prob, lambda_

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
    
    Returns
    -------
    A tuple with the smoothed probability and the alpha estimate.
    '''
    
    globl = 0
    if sum_globals:
        globl = global_freq / sum_globals
    
    prob = (local_freq + lambda_ * globl) /  (sum_locals + lambda_)
    alpha = lambda_ / (sum_locals + lambda_)
    return prob, alpha

def none(local_freq, sum_locals, global_freq, sum_globals, lambda_=0):
    '''
    Calculates the none smoothing for the given parameters. This
    is just JM with lambda = 0. The lambda_ parameter is ignore, it
    is here just to maintain the abstraction.
    
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
    
    Returns
    -------
    A tuple with the smoothed probability and the alpha estimate.
    '''
    return jelinek_mercer(local_freq, sum_locals, global_freq, sum_globals, 0)