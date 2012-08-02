# -*- coding: utf8
# cython: boundscheck = False
# cython: wraparound = False
'''This modules defines the base class which decorates other estimators'''

cimport cython
cimport numpy as np
np.import_array()

cdef np.ndarray EMPTY_RV = None

cdef class ProbabilityEstimator:
    '''
    Base class for probability estimates. This class only defines the methods
    to be implemented by subclasses. 
    '''

    cdef np.ndarray[np.float_t, ndim=1] prob_items_given_user(self, int user, 
            np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u), i.e., returns an array with the probability of each
        item given the user.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        Arguments
        ---------
        user: int
            User id
        gamma_items:
            Items to consider. 
        '''
        
        return EMPTY_RV

    cdef np.ndarray[np.float_t, ndim=1] prob_items_given_user_tag(self,
            int user, int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u,t), i.e., returns an array with the probability of each
        item given the user and the tag.
         
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        Arguments
        ---------
        user: int
            User id
        tag: int
            Tag id
        gamma_items:
            Items to consider. 
        '''
        return EMPTY_RV
    
    cdef np.ndarray[np.float_t, ndim=1] prob_items_given_tag(self, 
            int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|t), i.e., returns an array with the probability of each
        item given the tag.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        Arguments
        ---------
        tag: int
            User id
        gamma_items:
            Items to consider. 
        '''
        return EMPTY_RV
    
    cdef np.ndarray[np.float_t, ndim=1] prob_items(self, 
           np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I), i.e., returns an array with the probability of each
        item.

        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.

        Arguments
        ---------
        gamma_items:
            Items to consider.
        '''
        return EMPTY_RV

    cdef int num_tags(self):
        '''Get's the total number of tags'''
        return 0
    
cdef class DecoratorEstimator:
    '''
    This decorator provides the bridge between cython implementations and
    python code. This class is mostly used for unit testing since the 
    probability estimators are accessed from Cython code.
    '''
    
    cdef ProbabilityEstimator decorated_estimator
    
    def __init__(self, ProbabilityEstimator to_decorate):
        self.decorated_estimator = to_decorate

    def prob_items_given_user(self, int user,
                               np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u), i.e., returns an array with the probability of each
        item given the user.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        Arguments
        ---------
        user: int
            User id
        gamma_items:
            Items to consider. 
        '''
        return self.decorated_estimator.prob_items_given_user(user, gamma_items)

    def prob_items_given_user_tag(self,
                                  int user, int tag, 
                                  np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|u,t), i.e., returns an array with the probability of each
        item given the user and the tag.
         
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        Arguments
        ---------
        user: int
            User id
        tag: int
            Tag id
        gamma_items:
            Items to consider. 
        '''
        return self.decorated_estimator.prob_items_given_user_tag(user, tag,
                                                                  gamma_items)
    
    def prob_items_given_tag(self, int tag, 
                             np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|t), i.e., returns an array with the probability of each
        item given the tag.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        Arguments
        ---------
        tag: int
            User id
        gamma_items:
            Items to consider. 
        '''
        return self.decorated_estimator. prob_items_given_tag(tag, gamma_items)
    
    def prob_items(self, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I), i.e., returns an array with the probability of each
        item.

        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.

        Arguments
        ---------
        gamma_items:
            Items to consider.
        '''
        return self.decorated_estimator.prob_items(gamma_items)
    
    def num_tags(self):
        '''Get's the total number of tags'''
        return self.decorated_estimator.num_tags()