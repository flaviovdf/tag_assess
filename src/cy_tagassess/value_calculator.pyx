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
        Creates a map for the relevance of each item to the given user.
        
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
        Creates a map for the value of each tag to the given user.
        
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

        cdef np.ndarray[np.float64_t, ndim=1] vp_i
        cdef np.ndarray[np.float64_t, ndim=1] vp_ui
        cdef np.ndarray[np.float64_t, ndim=1] vp_ti
        cdef np.ndarray[np.float64_t, ndim=1] vp_iu
        cdef np.ndarray[np.float64_t, ndim=1] vp_itu
            
        est = self.est
        
        vp_i = est.vect_prob_item(items)
        vp_ui = est.vect_prob_user_given_item(items, user)
        p_u = est.prob_user(user) 
        
        #Computation
        return_val = {}
        for tag in range(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                continue
                
            vp_ti = est.vect_prob_tag_given_item(items, tag)
            
            vp_iu = vp_ui * (vp_i / p_u)
            vp_itu = vp_ti * vp_ui * (vp_i / (p_u * p_t))
            
            #Renormalization is necessary
            vp_iu /= vp_iu.sum()
            vp_itu /= vp_itu.sum()
            
            tag_val = entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            return_val[tag] = tag_val
        return return_val
    
    def tag_value_gcontext(self, 
                           np.ndarray[np.int64_t, ndim=1] gamma_items = None):
        '''
        Creates a map for the value of each tag in a global context.
         
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

        cdef np.ndarray[np.float64_t, ndim=1] vp_i
        cdef np.ndarray[np.float64_t, ndim=1] vp_ti
        cdef np.ndarray[np.float64_t, ndim=1] vp_it
        
        est = self.est
        vp_i = est.vect_prob_item(items)
        
        #Computation
        return_val = {}
        for tag in range(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                continue
                
            vp_ti = est.vect_prob_tag_given_item(items, tag)
            
            vp_it = vp_ti * (vp_i / p_t)
            
            #Renormalization is necessary
            vp_it /= vp_it.sum()
            vp_i /= vp_i.sum()
            
            tag_val = entropy.kullback_leiber_divergence(vp_it, vp_i)
            return_val[tag] = tag_val
        return return_val

    def mean_prob_item_given_user(self, int user, 
                                  np.ndarray[np.int64_t, ndim=1] items):
        '''
        Computes the average of P(I|u)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef double p_u
        cdef np.ndarray[np.float64_t, ndim=1] vp_i
        cdef np.ndarray[np.float64_t, ndim=1] vp_ui
        
        p_u = self.est.prob_user(user)
        vp_i = self.est.vect_prob_item(items)
        vp_ui = self.est.vect_prob_user_given_item(items, user)
        
        vp_iu = vp_ui * (vp_i / p_u)
        return vp_iu.mean()
    
    def mean_prob_item(self, np.ndarray[np.int64_t, ndim=1] items):
        '''
        Computes the average of P(I)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        
        cdef np.ndarray[np.float64_t, ndim=1] vp_i
        vp_i = self.est.vect_prob_item(items)
        return vp_i.mean()