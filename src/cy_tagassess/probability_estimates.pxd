# -*- coding: utf8
import numpy as np
cimport numpy as np

cdef class SmoothEstimator:
    
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
    cdef np.float64_t* item_col_mle
    
    cdef np.ndarray _tag_col_freq
    cdef np.int64_t* tag_col_freq
    
    cdef np.ndarray _item_local_sums
    cdef np.int64_t* item_local_sums
    
    #Auxiliary dictionaries
    cdef dict item_tag_freq
    cdef dict user_tags 
    cdef int profile_size
    
    cpdef double prob_item(self, int item)

    cpdef double prob_tag(self, int tag)
    
    cpdef double prob_tag_given_item(self, int item, int tag)
    
    cpdef double prob_user(self, int user)
    
    cpdef double prob_user_given_item(self, int item, int user)
    
    cpdef double log_prob_tag(self, int tag)
    
    cpdef double log_prob_tag_given_item(self, int item, int tag)
    
    cpdef double log_prob_item(self, int item)

    cpdef double log_prob_user(self, int user)
    
    cpdef double log_prob_user_given_item(self, int item, int user)
    
    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_user(self, 
            np.ndarray[np.int64_t, ndim=1] users)

    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_item(self, 
            np.ndarray[np.int64_t, ndim=1] items)

    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_tag(self, 
            np.ndarray[np.int64_t, ndim=1] tags)

    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_user_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int user)
        
    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_tag_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int tag)
   
    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_user(self, 
            np.ndarray[np.int64_t, ndim=1] users)

    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_item(self, 
            np.ndarray[np.int64_t, ndim=1] items)

    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_tag(self, 
            np.ndarray[np.int64_t, ndim=1] tags)

    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_user_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int user)
    
    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_tag_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int tag)
   
    cpdef double tag_pop(self, int tag)

    cpdef double item_tag_pop(self, int item, int tag)