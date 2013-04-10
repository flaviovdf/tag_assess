# -*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''Probability based on smoothing methods'''

from __future__ import division, print_function

from collections import defaultdict

from tagassess.probability_estimates.smooth cimport bayes
from tagassess.probability_estimates.smooth cimport jelinek_mercer

import heapq
import numpy as np

cimport base
cimport numpy as np
np.import_array()

cdef int JM = 1
cdef int BAYES = 2

cdef class SmoothEstimator(base.ProbabilityEstimator):
    '''
    Implementation of a similar approach as proposed in:
    
    Personalization of Tagging Systems, 
    Wang, Jun, Clements Maarten, Yang J., de Vries Arjen P., and 
    Reinders Marcel J. T. , 
    Information Processing and Management, Volume 46, Issue 1, p.58-70, (2010)
    
    In details:
        * P(i) = Base on MLE.
        * P(t|i) = P(t|M_i)$ where, $M_i$ is a smoothed model of the items
        * P(t) = Sum of P(t|i) * P(i) for every item
        * P(u) and P(u|i) considers users as tags. More specifically, the past
          tags used by the user. So, these two functions will make use of $P(t)$
           and $P(t|i)$.
    '''

    def __init__(self, smooth_method, lambda_, annotation_it, 
                 user_profile_size = -1):
        super(SmoothEstimator, self).__init__()
        
        smooths = {'JM':JM,
                   'Bayes':BAYES}
        
        self.n_annotations = 0
        self.smooth_func_id = smooths[smooth_method]
        self.lambda_ = lambda_
        
        self.item_tag_freq = {}
        self.user_tags = {}
        self.profile_size = user_profile_size
        
        self.__populate(annotation_it)
        
    def __populate(self, annotation_it):
        '''
        Computes initial indexes based on the iterator
        
        Arguments
        ---------
        annotation_it: iterable
            An iterable with annotations
        '''
        tag_col_dict = defaultdict(int)
        item_col_dict = defaultdict(int)
        item_tag_dict = defaultdict(lambda: defaultdict(int))
        user_tags_dict = defaultdict(lambda: defaultdict(int))
        
        max_tag = 0
        max_item = 0
        
        #For this class we need user and item indexes
        self.n_annotations = 0
        for annotation in annotation_it:
            self.n_annotations += 1
            
            #Initial updates
            tag = annotation['tag']
            item = annotation['item']
            user = annotation['user']
            
            tag_col_dict[tag] += 1
            item_col_dict[item] += 1
            item_tag_dict[item][tag] += 1
            user_tags_dict[user][tag] += 1
            
            #Tag, item and user id space being defined            
            if tag > max_tag:
                max_tag = tag
                
            if item > max_item:
                max_item = item
        
        #Initializing arrays
        self.n_tags = max_tag + 1
        self.n_items = max_item + 1
        self.n_users = len(user_tags_dict)
        
        self.tag_col_freq = np.zeros(self.n_tags, dtype='i')
        for tag in tag_col_dict:
            self.tag_col_freq[tag] = tag_col_dict[tag]
        
        self.item_col_mle = np.zeros(self.n_items, dtype='f')
        self.item_local_sums = np.zeros(self.n_items, dtype='i')
        
        for item in item_col_dict:
            self.item_col_mle[item] = item_col_dict[item] / self.n_annotations
            
            sum_local = sum(item_tag_dict[item].values())
            self.item_local_sums[item] = sum_local
            
            for tag in item_tag_dict[item]:
                self.item_tag_freq[item, tag] = item_tag_dict[item][tag]
        
        #User profile
        for user in user_tags_dict:
            tags = [(freq, tag) for tag, freq in user_tags_dict[user].items()]
            if self.profile_size == -1 or self.profile_size > len(tags):
                aux = tags
            else:
                aux = heapq.nlargest(self.profile_size, tags)
            
            self.user_tags[user] = np.array([tag[1] for tag in aux])
    
    cpdef double prob_item(self, int item):
        '''Probability of seeing a given item. $P(i)$'''
        
        if item < 0 or item >= self.n_items:
            return 0.0
        
        return self.item_col_mle[item]

    cpdef double prob_tag_given_item(self, int item, int tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        
        if item < 0 or item >= self.n_items:
            return 0.0

        if tag < 0 or tag >= self.n_tags:
            return 0.0
                
        cdef object key = (item, tag)
        cdef int local_freq = 0

        if key in self.item_tag_freq:
            local_freq = <int> self.item_tag_freq[key]
            
        cdef int sum_local = self.item_local_sums[item]
        cdef double prob
        
        if self.smooth_func_id == JM:
            prob = jelinek_mercer(local_freq,
                                   sum_local,
                                   self.tag_col_freq[tag],
                                   self.n_annotations,
                                   self.lambda_)
        elif self.smooth_func_id == BAYES:
            prob = bayes(local_freq,
                         sum_local,
                         self.tag_col_freq[tag],
                         self.n_annotations,
                         self.lambda_)
        else:
            print('Unknown smooth')
            return 0.0
        
        return prob
    
    cpdef double prob_user_given_item(self, int item, int user):
        '''Probability of seeing an user given an item. $P(u|i)$'''

        if item < 0 or item >= self.n_items:
            return 0.0

        if user < 0 or user >= self.n_users:
            return 0.0

        cdef np.ndarray[np.int_t, ndim=1] utags = \
                self.user_tags[user]
                
        cdef double return_val = 1.0
        cdef Py_ssize_t i
        for i in range(utags.shape[0]):
            return_val *= self.prob_tag_given_item(item, utags[i])
        return return_val
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user(self, int user, 
            np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u), i.e., returns an array with the probability of each
        item given the user.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.

        In this estimator, we compute this probability as:
        
        .. math::
            p(i|u) & =       & p(u|i)p(i) / p(u) \\
                   & \propto & p(u|i)p(i)
        
        p(u|i) considers users as a query composed of her past tags.
        Thus, p(u) is the product p(t|i) for every tag used by the user.
          
        Arguments
        ---------
        user: int
            User id
        gamma_items:
            Items to consider. 
        '''
        
        cdef Py_ssize_t n_items = gamma_items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] vp_iu = np.ndarray(n_items)
        cdef double sum_probs = 0
        
        cdef Py_ssize_t item
        for item from 0 <= item < n_items:
            vp_iu[item] = self.prob_user_given_item(item, user) * \
                          self.prob_item(item)
            sum_probs += vp_iu[item]
            
        for item from 0 <= item < n_items:
            vp_iu[item] = vp_iu[item] / sum_probs

        return vp_iu

    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user_tag(self,
            int user, int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u,t), i.e., returns an array with the probability of each
        item given the user and the tag.
         
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        In this estimator, we compute this probability as:
        
        .. math::
            p(i|t,u) & =       & p(i,t,u) / p(t,u) \\
                     & =       & p(u|i)p(t|i)p(i) / p(t,u) \\
                     & \propto & p(u|i)p(t|i)p(i)
        
        Arguments
        ---------
        user: int
            User id
        tag: int
            Tag id
        gamma_items:
            Items to consider. 
        '''
        cdef Py_ssize_t n_items = gamma_items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] vp_itu = np.zeros(n_items)
        cdef double sum_probs = 0
        
        cdef Py_ssize_t item
        for item from 0 <= item < n_items:
            vp_itu[item] = self.prob_user_given_item(item, user) * \
                           self.prob_tag_given_item(item, tag) * \
                           self.prob_item(item)
            
            sum_probs += vp_itu[item]

        for item from 0 <= item < n_items:
            vp_itu[item] = vp_itu[item] / sum_probs

        return vp_itu
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_tag(self, 
            int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|t), i.e., returns an array with the probability of each
        item given the tag.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        On this estimator p(i | t) is proportional to:
        
        ..math:: 
            p(i | t) \propto  p(t | i)  * p(i)
        
        p(t | i) comes from the smoothing method, while p(i) is based on item
        frequency.
        
        Arguments
        ---------
        tag: int
            User id
        gamma_items:
            Items to consider. 
        '''
        cdef Py_ssize_t n_items = gamma_items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] vp_it = np.ndarray(n_items)
        cdef double sum_probs = 0
        
        cdef int item
        for item from 0 <= item < n_items:
            vp_it[item] = self.prob_tag_given_item(item, tag) * \
                          self.prob_item(item)
            sum_probs += vp_it[item]
        
        for item from 0 <= item < n_items:
            vp_it[item] = vp_it[item] / sum_probs

        return vp_it
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items(self, 
           np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I), i.e., returns an array with the probability of each
        item.

        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.

        For this estimator this probability is proportional to the frequency
        of the item on the dataset. 
        
        ..math:: 
            p(i) = N_i / N

        Arguments
        ---------
        gamma_items:
            Items to consider.
        '''
        
        cdef Py_ssize_t n_items = gamma_items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] vp_i = np.ndarray(n_items)
        cdef double sum_probs = 0
        
        cdef Py_ssize_t item
        for item from 0 <= item < n_items:
            vp_i[item] = self.prob_item(item)
            sum_probs += vp_i[item]
        
        for item from 0 <= item < n_items:
            vp_i[item] = vp_i[item] / sum_probs

        return vp_i
    
    cpdef int num_tags(self):
        return self.n_tags