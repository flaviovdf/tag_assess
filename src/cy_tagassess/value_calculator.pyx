# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from cy_tagassess cimport entropy
from tagassess.recommenders import Recommender

import numpy as np
cimport numpy as np

cdef class ValueCalculator(object):
    '''
    Class used to compute values. 
    Contains basic value functions and filtering.
    '''
    cdef object est
    cdef object recc
    
    def __init__(self, object estimator, object recommender):
        self.est = estimator
        self.recc = recommender
        
    def item_value(self, int user):
        '''
        Creates a generator for the relevance of each item to the given user.
        The generator will yield the tuple: (item_relevance, item).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef Py_ssize_t i = 0
        
        return_val = {}
        for i in range(self.est.num_items()):
            relevance = self.recc.relevance(user, i)
            return_val[i] = relevance
        return return_val

    def tag_value_ucontext(self, int user, 
                           np.ndarray[np.int64_t, ndim=1] gamma_items = None):
        '''
        Creates a generator for the value of each tag to the given user.
        The generator will yield the tuple: (tag_value, tag).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.int64_t, ndim=1] items
        if gamma_items is not None:
            items = gamma_items
        else:
            items = np.arange(self.est.num_items())

        #Variables
        cdef Py_ssize_t tag = 0
        
        cdef double p_u
        cdef double p_t

        cdef np.ndarray[np.float64_t, ndim=1] p_i
        cdef np.ndarray[np.float64_t, ndim=1] p_ui
        cdef np.ndarray[np.float64_t, ndim=1] p_ti
        cdef np.ndarray[np.float64_t, ndim=1] p_iu
        cdef np.ndarray[np.float64_t, ndim=1] p_itu
            
        est = self.est
        
        p_i = est.vect_prob_item(items)
        p_ui = est.vect_prob_user_given_item(items, user)
        p_u = est.prob_user(user) 
        
        #Computation
        return_val = {}
        for tag in range(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                continue
                
            p_ti = est.vect_prob_tag_given_item(items, tag)
            
            p_iu = p_ui * p_i / p_u
            p_itu = p_ti * p_ui * p_i / (p_u * p_t)
            
            #Renormalization is necessary
            p_iu /= p_iu.sum()
            p_itu /= p_itu.sum()
            
            tag_val = entropy.kullback_leiber_divergence(p_itu, p_iu)
            return_val[tag] = tag_val
        return return_val
    
    def tag_value_gcontext(self, 
                           np.ndarray[np.int64_t, ndim=1] gamma_items = None):
        '''
        Creates a generator for the value of each tag in a global context.
        The generator will yield the tuple: (tag_value, tag).
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        #Variables
        cdef np.ndarray[np.int64_t, ndim=1] items
        if gamma_items is not None:
            items = gamma_items
        else:
            items = np.arange(self.est.num_items())
        
        cdef Py_ssize_t tag = 0                 
        cdef double p_t

        cdef np.ndarray[np.float64_t, ndim=1] p_i
        cdef np.ndarray[np.float64_t, ndim=1] p_ti
        cdef np.ndarray[np.float64_t, ndim=1] p_it
        
        est = self.est
        p_i = est.vect_prob_item(items)
        
        #Computation
        return_val = {}
        for tag in range(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                continue
                
            p_ti = est.vect_prob_tag_given_item(items, tag)
            
            p_it = p_ti * p_i / p_t
            
            #Renormalization is necessary
            p_it /= p_it.sum()
            p_i /= p_i.sum()
            
            tag_val = entropy.kullback_leiber_divergence(p_it, p_i)
            return_val[tag] = tag_val
        return return_val
