# -*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from tagassess cimport entropy
from tagassess.probability_estimates.base cimport ProbabilityEstimator

cimport cython
import numpy as np
cimport numpy as np

cdef class ValueCalculator(object):
    '''
    Class used to compute tag values. 
    Contains basic value functions and filtering.
    '''
    cdef ProbabilityEstimator est
    
    def __init__(self, ProbabilityEstimator estimator):
        self.est = estimator

    def tag_value_personalized(self, int user, 
            np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Creates an array for the value of each tag to the given user.
        In details, this computes:
        
        D( P(i | t, u) || P(i | u) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.float_t, ndim=1] return_val
        return_val = np.zeros(self.est.num_tags(), dtype='d')
       
        cdef np.ndarray[np.float_t, ndim=1] vp_iu
        cdef np.ndarray[np.float_t, ndim=1] vp_itu
        cdef double tag_val
        cdef Py_ssize_t tag
        for tag in range(return_val.shape[0]):
            vp_iu = self.est.prob_items_given_user(user, gamma_items)
            vp_itu = self.est.prob_items_given_user_tag(user, tag, 
                                                        gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            return_val[tag] = tag_val
        return return_val
    
    def tag_value_item_search(self, 
            np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Creates an array for the value of each tag in a global context.
        
        In details, this computes:
        
        D( P(i | t) || P(i) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.float_t, ndim=1] return_val
        return_val = np.zeros(self.est.num_tags(), dtype='d')
        
        cdef np.ndarray[np.float_t, ndim=1] vp_i = \
                self.est.prob_items(gamma_items)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_it
        cdef double tag_val
        cdef Py_ssize_t tag
        for tag in range(return_val.shape[0]):
            vp_it = self.est.prob_items_given_tag(tag, gamma_items)
            tag_val = entropy.kullback_leiber_divergence(vp_it, vp_i)
            return_val[tag] = tag_val
        return return_val

    def tag_value_per_user_search(self, int user, 
                                  np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Creates an array for the value of each tag in a global context.
        
        In details, this computes:
        
        D( P(i | t) || P(i | u) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.float_t, ndim=1] return_val
        return_val = np.zeros(self.est.num_tags(), dtype='d')
        
        cdef np.ndarray[np.float_t, ndim=1] vp_it
        cdef np.ndarray[np.float_t, ndim=1] vp_iu
        cdef double tag_val
        cdef Py_ssize_t tag
        
        for tag in range(return_val.shape[0]):
            vp_it = self.est.prob_items_given_tag(tag, gamma_items)
            vp_iu = self.est.prob_items_given_user(user, gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_iu, vp_it)
            return_val[tag] = tag_val
        return return_val
