# -*- coding: utf8
# cython: boundscheck = False
# cython: wraparound = False
'''This modules defines the base class which decorates other estimators'''

cimport cython
cimport numpy as np
np.import_array()

cdef class ProbabilityEstimator:
    '''
    Base class for probability estimates. This class only defines the methods
    to be implemented by subclasses. 
    '''

    cpdef np.ndarray[np.double_t, ndim=1] prob_items_given_user(self, int user, 
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
        pass

    cpdef np.ndarray[np.double_t, ndim=1] prob_items_given_user_tag(self,
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
        pass
    
    cpdef np.ndarray[np.double_t, ndim=1] prob_items_given_tag(self, 
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
        pass
    
    cpdef np.ndarray[np.double_t, ndim=1] prob_items(self, 
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
        pass
