# -*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from tagassess.index_creator import create_occurrence_index
from tagassess.stats.topk import kendall_tau_distance as dktau

import numpy as np

from tagassess cimport entropy
from tagassess.probability_estimates.base cimport ProbabilityEstimator

cimport cython
cimport numpy as np

cdef class ValueCalculator(object):
    '''
    Class used to compute tag values. 
    Contains basic value functions and filtering.
    '''
    
    cdef dict items_with_tag
    cdef ProbabilityEstimator est
    
    def __init__(self, ProbabilityEstimator estimator, object annotation_it):
        self.est = estimator
        self.items_with_tag = dict((k, v) for k, v in 
            create_occurrence_index(annotation_it, 'tag', 'item').items())

    cpdef calc_rho(self, int tag, 
            np.ndarray[np.int_t, ndim=1] top_valued_items):
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
        top_valued_items : int array
            Item reverse sorted according to p(i|u)
        
        See also
        --------
        tagassess.stats.topk
        '''
        
        #Populates I^t with top valued items
        cdef Py_ssize_t num_items_for_tag = len(self.items_with_tag[tag])
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
            return 1 / (1 + dktau(top_valued_items, top_valued_items_with_tag, 
                                  k, p=1))

    def tag_value_personalized(self, int user, 
            np.ndarray[np.int_t, ndim=1] gamma_items,
            np.ndarray[np.int_t, ndim=1] tags):
        '''
        Creates an array for the value of each tag to the given user.
        In details, this computes:
        
        Rho * D( P(i | t, u) || P(i | u) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.float_t, ndim=1] return_val
        return_val = np.ndarray(tags.shape[0], dtype='d')
       
        cdef np.ndarray[np.float_t, ndim=1] vp_iu
        cdef np.ndarray[np.float_t, ndim=1] vp_itu
        cdef double tag_val
        cdef double rho 
        cdef Py_ssize_t i
        cdef int tag
        for i in range(tags.shape[0]):
            tag = tags[i]
            vp_iu = self.est.prob_items_given_user(user, gamma_items)
            vp_itu = self.est.prob_items_given_user_tag(user, tag, 
                                                        gamma_items)
            rho = self.calc_rho(tag, vp_iu.argsort()[::-1])
            tag_val = rho * entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            return_val[tag] = tag_val
        return return_val
    
    def tag_value_item_search(self, 
            np.ndarray[np.int_t, ndim=1] gamma_items,
            np.ndarray[np.int_t, ndim=1] tags):
        '''
        Creates an array for the value of each tag in a global context.
        
        In details, this computes:
        
        Rho * D( P(i | t) || P(i) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        cdef np.ndarray[np.float_t, ndim=1] return_val
        return_val = np.ndarray(tags.shape[0], dtype='d')
        
        cdef np.ndarray[np.float_t, ndim=1] vp_i = \
                self.est.prob_items(gamma_items)
        
        cdef np.ndarray[np.float_t, ndim=1] vp_it
        cdef double tag_val
        cdef double rho
        cdef Py_ssize_t i
        cdef int tag
        for i in range(tags.shape[0]):
            tag = tags[i]
            vp_it = self.est.prob_items_given_tag(tag, gamma_items)
            rho = self.calc_rho(tag, vp_i.argsort()[::-1])
            tag_val = rho * entropy.kullback_leiber_divergence(vp_it, vp_i)
            return_val[tag] = tag_val
        return return_val