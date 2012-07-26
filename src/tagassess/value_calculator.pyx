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
    Class used to compute values. 
    Contains basic value functions and filtering.
    '''
    cdef ProbabilityEstimator est
    cdef object recc
    
    def __init__(self, ProbabilityEstimator estimator):
        self.est = estimator

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def tag_value_personalized(self, int user, 
            np.ndarray[np.int_t, ndim=1] gamma_items = None):
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
            vp_iu = self.rnorm_prob_items_given_user(user, gamma_items)
            vp_itu = self.rnorm_prob_items_given_user_tag(user, tag, 
                                                          gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            return_val[tag] = tag_val
        return return_val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def tag_value_item_search(self, 
            np.ndarray[np.int_t, ndim=1] gamma_items = None):
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
                self.rnorm_prob_items(gamma_items)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_it
        cdef double tag_val
        cdef Py_ssize_t tag
        for tag in range(return_val.shape[0]):
            vp_it = self.rnorm_prob_items_given_tag(tag, gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_it, vp_i)
            return_val[tag] = tag_val
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def tag_value_per_user_search(self, user, gamma_items = None):
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
            vp_it = self.rnorm_prob_items_given_tag(tag, gamma_items)
            vp_iu = self.rnorm_prob_items_given_user(user, gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_iu, vp_it)
            return_val[tag] = tag_val
        return return_val
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float_t, ndim=1] rnorm_prob_items_given_user(self, int user, 
            np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
        
        cdef double p_u = self.est.prob_user(user)
        cdef np.ndarray[np.float_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)
        cdef np.ndarray[np.float_t, ndim=1] vp_ui = \
                self.est.vect_prob_user_given_item(items, user)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_iu = np.ndarray(items.shape[0])
        cdef Py_ssize_t i
        cdef double sum_probs = 0
        for i in range(items.shape[0]):
            vp_iu[i] = vp_ui[i] * vp_i[i] / p_u
            sum_probs += vp_iu[i]
        
        for i in range(items.shape[0]):
            vp_iu[i] = vp_iu[i] / sum_probs

        return vp_iu

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float_t, ndim=1] rnorm_prob_items_given_user_tag(self,
            int user, int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u,t)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
        
        cdef double p_t = self.est.prob_tag(tag)
        cdef double p_u = self.est.prob_user(user)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)
        cdef np.ndarray[np.float_t, ndim=1] vp_ui = \
                self.est.vect_prob_user_given_item(items, user)
        cdef np.ndarray[np.float_t, ndim=1] vp_ti = \
                self.est.vect_prob_tag_given_item(items, tag)

        cdef np.ndarray[np.float_t, ndim=1] vp_itu = \
                np.ndarray(items.shape[0])
        cdef Py_ssize_t i
        cdef double sum_probs = 0
        for i in range(items.shape[0]):
            vp_itu[i] = vp_ti[i] * vp_ui[i] * (vp_i[i] / (p_u * p_t))
            sum_probs += vp_itu[i]

        for i in range(items.shape[0]):
            vp_itu[i] = vp_itu[i] / sum_probs

        return vp_itu
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float_t, ndim=1] rnorm_prob_items_given_tag(self, 
            int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|t)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
        
        cdef double p_t = self.est.prob_tag(tag)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)
        cdef np.ndarray[np.float_t, ndim=1] vp_ti = \
                self.est.vect_prob_tag_given_item(items, tag)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_it = np.ndarray(items.shape[0])
        cdef Py_ssize_t i
        cdef double sum_probs = 0
        for i in range(items.shape[0]):
            vp_it[i] = vp_ti[i] * vp_i[i] / p_t 
            sum_probs += vp_it[i]
        
        for i in range(items.shape[0]):
            vp_it[i] = vp_it[i] / sum_probs

        return vp_it
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float_t, ndim=1] rnorm_prob_items(self, 
           np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
            
        cdef np.ndarray[np.float_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)

        cdef Py_ssize_t i
        cdef double sum_probs = sum(vp_i)
        for i in range(items.shape[0]):
            vp_i[i] = vp_i[i] / sum_probs

        return vp_i
