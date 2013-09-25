# -*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from tagassess.index_creator import create_occurrence_index
from tagassess.stats.topk import kendall_tau_distance as dktau


from cpython cimport bool
from tagassess cimport entropy
from tagassess.probability_estimates.base cimport ProbabilityEstimator

import numpy as np
cimport cython
cimport numpy as np

cdef class ValueCalculator(object):
    '''
    Class used to compute tag values. 
    Contains basic value functions and filtering.
    '''
    
    cdef dict items_with_tag
    cdef ProbabilityEstimator est
    cdef int num_items
    
    def __init__(self, ProbabilityEstimator estimator, object annotation_it):
        self.est = estimator
        self.items_with_tag = {}
        self.num_items = 0
        
        index = create_occurrence_index(annotation_it, 'tag', 'item').items()
        for k, v in index:
            self.num_items = max(self.num_items, max(v))
            self.items_with_tag[k] = v

    cpdef calc_rho(self, int tag, 
            np.ndarray[np.float_t, ndim=1] item_relevance,
            np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes rho for a given user and tag. Rho is given by the generalized 
        kendall distance with penalty between the items retrieved by the tag 
        sorted by p(i|u) and the items retrieved by the tag also sorted by 
        p(i|u).
        
        TODO: Although this method is inside a cythonized module,
        the it is not optimized. It is basically a drop in of pure python
        with some types added.
        
        Arguments
        ---------
        tag : int
            tag to compute rho
        item_relevance : float array
            Values of relevance for each item (p(i|u) or p(i|t,u))
        gamma_items : int array
            IDs of gamma items
        
        See also
        --------
        tagassess.stats.topk
        '''
        
        cdef np.ndarray[np.int_t, ndim=1] top_valued_items = \
                gamma_items[item_relevance.argsort()[::-1]]
        
        #Populates I^t with top valued items
        cdef list top_valued_items_with_tag = []
        
        cdef Py_ssize_t i = 0
        cdef int item_id
        for i in range(top_valued_items.shape[0]):
            item_id = top_valued_items[i] #Top valued items is already sorted
            if item_id in self.items_with_tag[tag]:
                top_valued_items_with_tag.append(item_id)
        
        cdef Py_ssize_t k = len(top_valued_items_with_tag)
        if k == 0:
            return 0
        else:
            return 1 - dktau(top_valued_items, top_valued_items_with_tag, k,
                             p=1)

    def tag_value_naive(self, int user, 
            np.ndarray[np.int_t, ndim=1] tags,
            bool return_rho_dkl=False):
        
        if not return_rho_dkl:
            return_val = np.ndarray(shape=(tags.shape[0],), dtype='d')
        else:
            return_val = np.ndarray(shape=(tags.shape[0], 3), dtype='d')
            
        cdef np.ndarray[np.float_t, ndim=1] vp_iu
        vp_iu = self.est.prob_items_given_user(user, np.arange(self.num_items))
        
        cdef double reduction = 0
        cdef double relevance = 0
        cdef int num_retrieved = 0
        
        cdef int item
        cdef int tag
        cdef Py_ssize_t i
        for i in range(tags.shape[0]):
            tag = tags[i]
            reduction = 0
            relevance = 0
            num_retrieved = 0
            for item in self.items_with_tag[tag]:
                relevance += vp_iu[item]
                num_retrieved += 1
            
            reduction = self.num_items - num_retrieved
            relevance /= num_retrieved
            if not return_rho_dkl:
                return_val[i] = reduction * relevance
            else:
                return_val[i, 0] = reduction
                return_val[i, 1] = relevance
                return_val[i, 2] = reduction * relevance
        return return_val
                
    def tag_value_personalized(self, int user, 
            np.ndarray[np.int_t, ndim=1] gamma_items,
            np.ndarray[np.int_t, ndim=1] tags,
            bool return_rho_dkl=False):
        '''
        Creates an array for the value of each tag to the given user.
        In details, this computes:
        
        Rho * D( P(i | t, u) || P(i | u) ),
        
        where D is the kullback-leiber divergence.
        
        If `return_rho_dkl` of the tag value wil be returned, else
        a matrix of size (len(tags), 3) will be returned. The first
        column will be rho values, the second dkl values and third will be
        the actual tag value rho * dkl.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if not return_rho_dkl:
            return_val = np.ndarray(shape=(tags.shape[0],), dtype='d')
        else:
            return_val = np.ndarray(shape=(tags.shape[0], 3), dtype='d')
       
        cdef np.ndarray[np.float_t, ndim=1] vp_iu
        cdef np.ndarray[np.float_t, ndim=1] vp_itu
        cdef double tag_val
        cdef double rho 
        cdef double dkl
        cdef Py_ssize_t i
        cdef int tag
        for i in range(tags.shape[0]):
            tag = tags[i]
            vp_iu = self.est.prob_items_given_user(user, gamma_items)
            vp_itu = self.est.prob_items_given_user_tag(user, tag, gamma_items)
            rho = self.calc_rho(tag, vp_iu, gamma_items)
            dkl = entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            tag_val = rho * dkl
            
            if not return_rho_dkl:
                return_val[i] = tag_val
            else:
                return_val[i, 0] = rho
                return_val[i, 1] = dkl
                return_val[i, 2] = tag_val
        return return_val
    
    def tag_value_item_search(self, 
            np.ndarray[np.int_t, ndim=1] gamma_items,
            np.ndarray[np.int_t, ndim=1] tags,
            bool return_rho_dkl=False):
        '''
        Creates an array for the value of each tag in a global context.
        
        In details, this computes:
        
        Rho * D( P(i | t) || P(i) ),
        
        where D is the kullback-leiber divergence.
        
        If `return_rho_dkl` of the tag value wil be returned, else
        a matrix of size (len(tags), 3) will be returned. The first
        column will be rho values, the second dkl values and third will be
        the actual tag value rho * dkl.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray return_val
        if not return_rho_dkl:
            return_val = np.ndarray(shape=(tags.shape[0],), dtype='d')
        else:
            return_val = np.ndarray(shape=(tags.shape[0], 3), dtype='d')
        
        cdef np.ndarray[np.float_t, ndim=1] vp_i = \
                self.est.prob_items(gamma_items)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_it
        cdef double tag_val
        cdef double rho
        cdef double dkl
        cdef Py_ssize_t i
        cdef int tag
        for i in range(tags.shape[0]):
            tag = tags[i]
            vp_it = self.est.prob_items_given_tag(tag, gamma_items)
            rho = self.calc_rho(tag, vp_i, gamma_items)
            dkl = entropy.kullback_leiber_divergence(vp_it, vp_i)
            tag_val = rho * dkl
            
            if not return_rho_dkl:
                return_val[i] = tag_val
            else:
                return_val[i, 0] = rho
                return_val[i, 1] = dkl
                return_val[i, 2] = tag_val
        return return_val