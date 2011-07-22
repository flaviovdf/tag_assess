# -*- coding: utf8
'''Estimator based on smoothing methods'''

from __future__ import division, print_function

from collections import defaultdict

from cy_tagassess.smooth cimport bayes
from cy_tagassess.smooth cimport jelinek_mercer

import heapq
import math
import numpy as np

cimport cython
cimport numpy as np

#Log2 from C99
cdef extern from "math.h":
    double log2(double)

cdef int JM = 1
cdef int BAYES = 2

cdef class SmoothEstimator:
    '''
    Implementation of the approach proposed in:
    
    Personalization of Tagging Systems, 
    Wang, Jun, Clements Maarten, Yang J., de Vries Arjen P., and Reinders Marcel J. T. , 
    Information Processing and Management, Volume 46, Issue 1, p.58-70, (2010)
    
    In details:
        * $P(t) and P(i) = Base on MLE.
        * $P(t|i) = P(t|M_i)$ where, $M_i$ is a smoothed model of the items
        * $P(u)$ and $P(u|i)$ considers users as tags. More specifically, the past
          tags used by the user. So, these two functions will make use of $P(t)$ and $P(t|i)$.
    '''
    
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

    def __init__(self, smooth_method, lambda_, annotation_it, user_profile_size = -1):
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
            
            #Tag, item and user id spaced being defined            
            if tag > max_tag:
                max_tag = tag
                
            if item > max_item:
                max_item = item
        
        #Initializing arrays
        self.n_tags = max_tag + 1
        self.n_items = max_item + 1
        self.n_users = len(user_tags_dict)
        
        self._tag_col_freq = np.zeros(self.n_tags)
        self.tag_col_freq = <np.int64_t*> self._tag_col_freq.data
        for tag in tag_col_dict:
            self.tag_col_freq[tag] = tag_col_dict[tag]
        
        self._item_col_mle = np.zeros(self.n_items, dtype='d')
        self.item_col_mle = <np.float64_t*> self._item_col_mle.data
        
        self._item_local_sums = np.zeros(shape = self.n_items)
        self.item_local_sums = <np.int64_t*> self._item_local_sums.data
        
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
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double prob_item(self, int item):
        '''Probability of seeing a given item. $P(i)$'''
        
        if item < 0 or item >= self.n_items:
            return 0.0
        
        return self.item_col_mle[item]

    @cython.boundscheck(False)
    @cython.wraparound(False)    
    cpdef double prob_tag(self, int tag):
        '''Probability of seeing a given tag. $P(t)$'''

        if tag < 0 or tag >= self.n_tags:
            return 0.0
        
        cdef double return_val = 0.0
        cdef double prob_tag_item
        
        for item in range(self.n_items):
            prob_tag_item = self.prob_tag_given_item(item, tag)
            return_val += self.item_col_mle[item] * prob_tag_item
        return return_val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
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
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double prob_user(self, int user):
        '''Probability of seeing an user. $P(u)$'''
        
        if user < 0 or user >= self.n_users:
            return 0.0
        
        cdef np.ndarray[np.int64_t, ndim=1] utags = \
                self.user_tags[user]

        cdef Py_ssize_t i
        cdef double return_val = 1.0
        for i in range(utags.shape[0]):
            return_val *= self.prob_tag(utags[i])
        return return_val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double prob_user_given_item(self, int item, int user):
        '''Probability of seeing an user given an item. $P(u|i)$'''

        if item < 0 or item >= self.n_items:
            return 0.0

        if user < 0 or user >= self.n_users:
            return 0.0

        cdef np.ndarray[np.int64_t, ndim=1] utags = \
                self.user_tags[user]
        
        cdef Py_ssize_t i
        cdef double return_val = 1.0
        for i in range(utags.shape[0]):
            return_val *= self.prob_tag_given_item(item, utags[i])
        return return_val
    
    #Log methods
    cpdef double log_prob_tag(self, int tag):
        '''Log probability of seeing a given tag. $P(t)$'''
        return log2(self.prob_tag(tag))
    
    cpdef double log_prob_tag_given_item(self, int item, int tag):
        '''Log probability of seeing a given tag for an item. $P(t|i)$'''
        return log2(self.prob_tag_given_item(item, tag))
    
    cpdef double log_prob_item(self, int item):
        '''Log probability of seeing a given item. $P(i)$'''
        return log2(self.prob_item(item))

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double log_prob_user(self, int user):
        '''
        Log probability of seeing an user. $P(u)$
        This method is useful when `prob_user` underflows.
        '''
        cdef np.ndarray[np.int64_t, ndim=1] utags = \
                self.user_tags[user]

        cdef Py_ssize_t i
        cdef double return_val = 0.0
        for i in range(utags.shape[0]):
            return_val += self.log_prob_tag(utags[i])
        return return_val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef double log_prob_user_given_item(self, int item, int user):
        '''
        Log probability of seeing an user given an item. $P(u|i)$.
        This method is useful when `prob_user_given_item` underflows.
        '''
        cdef np.ndarray[np.int64_t, ndim=1] utags = \
                self.user_tags[user]
        
        cdef Py_ssize_t i
        cdef double return_val = 0.0
        for i in range(utags.shape[0]):
            return_val += self.log_prob_tag_given_item(item, utags[i])
        return return_val
   
    #Vectorized methods
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_user(self, 
            np.ndarray[np.int64_t, ndim=1] users):

        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(users.shape[0])
        
        cdef Py_ssize_t i
        for i in range(users.shape[0]):
            return_val[i] = self.prob_user(users[i])
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_item(self, 
            np.ndarray[np.int64_t, ndim=1] items):
        
        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(items.shape[0])
        
        cdef Py_ssize_t i
        for i in range(items.shape[0]):
            return_val[i] = self.prob_item(items[i])
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_tag(self, 
            np.ndarray[np.int64_t, ndim=1] tags):

        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(tags.shape[0])
        
        cdef Py_ssize_t i
        for i in range(tags.shape[0]):
            return_val[i] = self.prob_tag(tags[i])
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_user_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int user):
        
        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(items.shape[0])
        
        cdef Py_ssize_t i
        for i in range(items.shape[0]):
            return_val[i] = self.prob_user_given_item(items[i], user)
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_prob_tag_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int tag):
        
        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(items.shape[0])
        cdef Py_ssize_t i
        for i in range(items.shape[0]):
            return_val[i] = self.prob_tag_given_item(items[i], tag)
        return return_val
   
    #Log vectorized methods
    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_user(self, 
            np.ndarray[np.int64_t, ndim=1] users):

        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(users.shape[0])
        
        cdef Py_ssize_t i
        for i in range(users.shape[0]):
            return_val[i] = self.log_prob_user(users[i])
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_item(self, 
            np.ndarray[np.int64_t, ndim=1] items):
        
        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(items.shape[0])
        
        cdef Py_ssize_t i
        for i in range(items.shape[0]):
            return_val[i] = self.log_prob_item(items[i])
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_tag(self, 
            np.ndarray[np.int64_t, ndim=1] tags):

        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(tags.shape[0])
        
        cdef Py_ssize_t i
        for i in range(tags.shape[0]):
            return_val[i] = self.log_prob_tag(tags[i])
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_user_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int user):
        
        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(items.shape[0])
        
        cdef Py_ssize_t i
        for i in range(items.shape[0]):
            return_val[i] = self.log_prob_user_given_item(items[i], user)
        return return_val

    @cython.boundscheck(False)
    @cython.wraparound(False)
    cpdef np.ndarray[np.float64_t, ndim=1] vect_log_prob_tag_given_item(self,
            np.ndarray[np.int64_t, ndim=1] items, int tag):
        
        cdef np.ndarray[np.float64_t, ndim=1] return_val = \
                np.ndarray(items.shape[0])
        cdef Py_ssize_t i
        for i in range(items.shape[0]):
            return_val[i] = self.log_prob_tag_given_item(items[i], tag)
        return return_val
   
    #Other methods
    def  num_items(self):
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