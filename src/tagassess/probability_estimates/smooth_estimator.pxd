# -*- coding: utf8

cimport base

import numpy as np
cimport numpy as np

cdef class SmoothEstimator(base.ProbabilityEstimator):
    
    #Size of data
    cdef int n_annotations
    cdef int n_items
    cdef int n_tags
    cdef int n_users

    #Smooth params
    cdef int smooth_func_id
    cdef double lambda_
    
    #Arrays declared as pointers, ugly cython hack
    cdef np.ndarray _item_col_mle
    cdef np.float_t* item_col_mle
    
    cdef np.ndarray _tag_col_freq
    cdef np.int_t* tag_col_freq
    
    cdef np.ndarray _item_local_sums
    cdef np.int_t* item_local_sums
    
    #Auxiliary dictionaries
    cdef dict item_tag_freq
    cdef dict user_tags 
    cdef int profile_size
    
    cpdef double tag_pop(self, int tag)

    cpdef double item_tag_pop(self, int item, int tag)