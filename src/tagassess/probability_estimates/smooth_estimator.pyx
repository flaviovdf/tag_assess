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
        
        self._tag_col_freq = np.zeros(self.n_tags)
        self.tag_col_freq = <np.int_t*> self._tag_col_freq.data
        for tag in tag_col_dict:
            self.tag_col_freq[tag] = tag_col_dict[tag]
        
        self._item_col_mle = np.zeros(self.n_items, dtype='d')
        self.item_col_mle = <np.float_t*> self._item_col_mle.data
        
        self._item_local_sums = np.zeros(shape = self.n_items)
        self.item_local_sums = <np.int_t*> self._item_local_sums.data
        
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
    
    cdef double prob_item(self, int item):
        '''Probability of seeing a given item. $P(i)$'''
        
        if item < 0 or item >= self.n_items:
            return 0.0
        
        return self.item_col_mle[item]

    cdef double prob_tag(self, int tag):
        '''Probability of seeing a given tag. $P(t)$'''

        if tag < 0 or tag >= self.n_tags:
            return 0.0
       
        cdef double return_val = 0.0
        cdef Py_ssize_t item
        for item in range(self.n_items):
            return_val += self.item_col_mle[item] * \
                          self.prob_tag_given_item(item, tag)

        return return_val
    
    cdef double prob_tag_given_item(self, int item, int tag):
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
    
    cdef double prob_user(self, int user):
        '''Probability of seeing an user. $P(u)$'''
        
        if user < 0 or user >= self.n_users:
            return 0.0
        
        cdef np.ndarray[np.int_t, ndim=1] utags = \
                self.user_tags[user]

        cdef double return_val = 1.0
        cdef Py_ssize_t i
        for i in range(utags.shape[0]):
            return_val *= self.prob_tag(utags[i])
        return return_val
    
    cdef double prob_user_given_item(self, int item, int user):
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
    
    #Other methods
    cpdef double tag_pop(self, int tag):
        '''Returns the popularity of a tag'''
        if tag >= self.n_tags or tag < 0:
            return 0
            
        return self.tag_col_freq[tag]

    cpdef double item_tag_pop(self, int item, int tag):
        '''Returns the popularity of a tag on an item'''
        if (item, tag) not in self.item_tag_freq:
            return 0
        return self.item_tag_freq[item, tag]
   
    def num_items(self):
        '''Number of items'''
        return self.n_items
    
    def num_tags(self):
        '''Number of tags'''
        return self.n_tags
    
    def num_users(self):
        '''Number of users'''
        return self.n_users
    
    def num_annotations(self):
        '''Number of annotations'''
        return self.n_annotations