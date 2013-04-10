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
    
    #These variables are cython memoryviews, think of them as 
    #arrays, you can do memview[a, b].
    cdef double[::1] item_col_mle
    cdef int[::1] tag_col_freq
    cdef int[::1] item_local_sums
    
    #Auxiliary dictionaries
    cdef dict item_tag_freq
    cdef dict user_tags 
    cdef int profile_size
 
    cpdef double prob_item(self, int item)

    cpdef double prob_tag_given_item(self, int item, int tag)
    
    cpdef double prob_user_given_item(self, int item, int user) 