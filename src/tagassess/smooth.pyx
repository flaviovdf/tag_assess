# -*- coding: utf8
#This is a Cython file. Hopefully, it will be the only one in the project.
#
#Due to the simplicity of the module, the code should be clear. The notes bellow can help
#with the different function types.
#
#Notes: 
# (1) cpdef makes the function a C function with the ability of it being called in
# python. 
# (2) cdef is a C function which can only be called by others (defined by cdef or cpdef).
# (3)normal functions are defined as def.
'''Implementation of the Jelinek-Mercer and Bayes smoothing.'''

from __future__ import division, print_function

cpdef jelinek_mercer(int local_freq, int sum_locals, int global_freq, 
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
    
    Returns
    -------
    A tuple with the smoothed probability and the alpha estimate.
    '''
    cdef double local = 0
    if sum_locals >= 0:
        local = (1 - lambda_) * local_freq / sum_locals
    
    cdef double globl = 0
    if sum_globals >= 0:
        globl = lambda_ * global_freq / sum_globals
    
    cdef double prob = local + globl
    return prob, lambda_

cpdef bayes(int local_freq, int sum_locals, int global_freq, 
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
    
    Returns
    -------
    A tuple with the smoothed probability and the alpha estimate.
    '''
    
    cdef double globl = 0
    cdef double cast_globl = global_freq
    if sum_globals >= 0:
        globl = cast_globl / sum_globals
    
    cdef double prob  = (local_freq + lambda_ * globl) /  (sum_locals + lambda_)
    cdef double alpha = lambda_ / (sum_locals + lambda_)
    return prob, alpha

def name_dict():
    '''Get's the smooth to name mappings'''
    smooths = {'JM':jelinek_mercer,
               'Bayes':bayes}
    return smooths
    
def get_by_name(name):
    '''
    Get's smooth function by name.
    
    Arguments
    ---------
    name: str {JM or Bayes}
    '''
    return name_dict()[name]
