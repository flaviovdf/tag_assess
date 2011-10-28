# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from cy_tagassess cimport entropy
from cy_tagassess.probability_estimates cimport SmoothEstimator

from tagassess.recommenders import Recommender

cimport cython
import numpy as np
cimport numpy as np

cdef class ValueCalculator(object):
    '''
    Class used to compute values. 
    Contains basic value functions and filtering.
    '''
    cdef SmoothEstimator est
    cdef object recc
    
    def __init__(self, SmoothEstimator estimator, object recommender):
        self.est = estimator
        self.recc = recommender

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def item_value(self, int user):
        '''
        Creates an array for the relevance of each item to the given user.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.float64_t, ndim=1] return_val
        return_val = np.zeros(self.est.num_items(), dtype='d')

        cdef Py_ssize_t item
        for item in range(return_val.shape[0]):
            relevance = self.recc.relevance(user, item)
            return_val[item] = relevance
        return return_val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def tag_value_personalized(self, int user, 
            np.ndarray[np.int64_t, ndim=1] gamma_items = None):
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
        cdef np.ndarray[np.float64_t, ndim=1] return_val
        return_val = np.zeros(self.est.num_tags(), dtype='d')
       
        cdef np.ndarray[np.float64_t, ndim=1] vp_iu
        cdef np.ndarray[np.float64_t, ndim=1] vp_itu
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
            np.ndarray[np.int64_t, ndim=1] gamma_items = None):
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
        cdef np.ndarray[np.float64_t, ndim=1] return_val
        return_val = np.zeros(self.est.num_tags(), dtype='d')
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_i = \
                self.rnorm_prob_items(gamma_items)
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_it
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
        cdef np.ndarray[np.float64_t, ndim=1] return_val
        return_val = np.zeros(self.est.num_tags(), dtype='d')
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_it
        cdef np.ndarray[np.float64_t, ndim=1] vp_iu
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
    cpdef np.ndarray[np.float64_t, ndim=1] rnorm_prob_items_given_user(self, int user, 
            np.ndarray[np.int64_t, ndim=1] gamma_items):
        '''
        Computes P(I|u)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int64_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
        
        cdef double p_u = self.est.prob_user(user)
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)
        cdef np.ndarray[np.float64_t, ndim=1] vp_ui = \
                self.est.vect_prob_user_given_item(items, user)
        
        vp_iu = vp_ui * (vp_i / p_u)
        vp_iu = vp_iu / vp_iu.sum()
        return vp_iu

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] rnorm_prob_items_given_user_tag(self,
            int user, int tag, np.ndarray[np.int64_t, ndim=1] gamma_items):
        '''
        Computes P(I|u,t)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int64_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
        
        cdef double p_t = self.est.prob_tag(tag)
        cdef double p_u = self.est.prob_user(user)
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)
        cdef np.ndarray[np.float64_t, ndim=1] vp_ui = \
                self.est.vect_prob_user_given_item(items, user)
        cdef np.ndarray[np.float64_t, ndim=1] vp_ti = \
                self.est.vect_prob_tag_given_item(items, tag)

        cdef np.ndarray[np.float64_t, ndim=1] vp_itu = \
                vp_ti * vp_ui * (vp_i / (p_u * p_t))
        
        vp_itu = vp_itu / vp_itu.sum()
        return vp_itu
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] rnorm_prob_items_given_tag(self, 
            int tag, np.ndarray[np.int64_t, ndim=1] gamma_items):
        '''
        Computes P(I|t)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int64_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
        
        cdef double p_t = self.est.prob_tag(tag)
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)
        cdef np.ndarray[np.float64_t, ndim=1] vp_ti = \
                self.est.vect_prob_tag_given_item(items, tag)
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_it = \
                vp_ti * (vp_i / p_t)

        vp_it = vp_it / vp_it.sum()
        return vp_it
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] rnorm_prob_items(self, 
           np.ndarray[np.int64_t, ndim=1] gamma_items):
        '''
        Computes P(I)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int64_t, ndim=1] items
        if gamma_items is None:
            items = np.arange(self.est.num_items())
        else:
            items = gamma_items
            
        cdef np.ndarray[np.float64_t, ndim=1] vp_i = \
                self.est.vect_prob_item(items)
        vp_i = vp_i / vp_i.sum()
        return vp_i
