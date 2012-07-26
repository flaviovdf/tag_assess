# -*- coding: utf8
# cython: boundscheck = False
# cython: wraparound = False
'''This modules defines the base class which decorates other estimators'''

cimport cython
cimport numpy as np
np.import_array()

cdef class ProbabilityEstimator:
    '''Base class for probability estimates'''

    cdef double prob_item(self, int item):
        '''Probability of seeing a given item. $P(i)$'''
        return 0

    cdef double prob_tag(self, int tag):
        '''Probability of seeing a given tag. $P(t)$'''
        return 0
    
    cdef double prob_tag_given_item(self, int item, int tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        return 0
    
    cdef double prob_user(self, int user):
        '''Probability of seeing an user. $P(u)$'''
        return 0
    
    cdef double prob_user_given_item(self, int item, int user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        return 0

    cdef np.ndarray[np.float_t, ndim=1] vect_prob_user(self, 
            np.ndarray[np.int_t, ndim=1] users):
        '''Computers the P(u) for a vector of users'''
        
        cdef Py_ssize_t n = users.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] return_val = np.ndarray(n)
        
        cdef Py_ssize_t i
        for i from 0 <= i < n:
            return_val[i] = self.prob_user(users[i])

        return return_val

    cdef np.ndarray[np.float_t, ndim=1] vect_prob_item(self, 
            np.ndarray[np.int_t, ndim=1] items):
        '''Computers the P(i) for a vector of items'''
        
        cdef Py_ssize_t n = items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] return_val = np.ndarray(n)
        
        cdef Py_ssize_t i
        for i from 0 <= i < n:
            return_val[i] = self.prob_item(items[i])

        return return_val

    cdef np.ndarray[np.float_t, ndim=1] vect_prob_tag(self, 
            np.ndarray[np.int_t, ndim=1] tags):
        '''Computers the P(t) for a vector of tags'''
        
        cdef Py_ssize_t n = tags.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] return_val = np.ndarray(n)
        
        cdef Py_ssize_t i
        for i from 0 <= i < n:
            return_val[i] = self.prob_tag(tags[i])

        return return_val

    cdef np.ndarray[np.float_t, ndim=1] vect_prob_user_given_item(self,
            np.ndarray[np.int_t, ndim=1] items, int user):
        '''Computers the P(u|i) for a vector of items'''
        
        cdef Py_ssize_t n = items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] return_val = np.ndarray(n)
        
        cdef Py_ssize_t i
        for i from 0 <= i < n:
            return_val[i] = self.prob_user_given_item(items[i], user)

        return return_val
        
    cdef np.ndarray[np.float_t, ndim=1] vect_prob_tag_given_item(self,
            np.ndarray[np.int_t, ndim=1] items, int tag):
        '''Computers the P(t|i) for a vector of items'''
        
        cdef Py_ssize_t n = items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] return_val = np.ndarray(n)
        
        cdef Py_ssize_t i
        for i from 0 <= i < n:
            return_val[i] = self.prob_tag_given_item(items[i], tag)

        return return_val

cdef class DecoratorEstimator:
    '''
    This decorator provides the bridge between cython implementations and
    python code. This class is mostly used for unit testing since the 
    probability estimators are accessed from Cython code.
    '''
    
    cdef ProbabilityEstimator decorated_estimator

    def __init__(self, ProbabilityEstimator to_decorate):
        self.decorated_estimator = to_decorate
    
    def prob_tag(self, int tag):
        return self.decorated_estimator.prob_tag(tag)
    
    def prob_tag_given_item(self, int item, int tag):
        return self.decorated_estimator.prob_tag_given_item(item, tag)
    
    def prob_user(self, int user):
        return self.decorated_estimator.prob_user(user)
    
    def prob_user_given_item(self, int item, int user):
        return self.decorated_estimator.prob_user_given_item(item, user)
    
    def prob_item(self, int item):
        return self.decorated_estimator.prob_item(item)

    def vect_prob_user(self, np.ndarray[np.int_t, ndim=1] users):
        return self.decorated_estimator.vect_prob_user(users)

    def vect_prob_item(self, np.ndarray[np.int_t, ndim=1] items):
        return self.decorated_estimator.vect_prob_item(items)

    def vect_prob_tag(self, np.ndarray[np.int_t, ndim=1] tags):
        return self.decorated_estimator.vect_prob_tag(tags)
    
    def vect_prob_user_given_item(self, np.ndarray[np.int_t, ndim=1] items, 
                                  int user):
        return self.decorated_estimator.vect_prob_user_given_item(items, user)
            
    def vect_prob_tag_given_item(self, np.ndarray[np.int_t, ndim=1] items, 
                                 int tag):
        return self.decorated_estimator.vect_prob_tag_given_item(items, tag)