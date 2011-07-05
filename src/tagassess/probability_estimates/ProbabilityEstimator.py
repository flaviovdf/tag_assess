# -*- coding: utf8
'''
Abstract base class for probability estimators
'''
from __future__ import division, print_function

import abc
import numpy as np

class ProbabilityEstimator(object):
    '''Base class for probability estimates'''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def prob_tag(self, tag):
        '''Probability of seeing a given tag. $P(t)$'''
        pass
    
    @abc.abstractmethod
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        pass
    
    @abc.abstractmethod
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        pass
    
    @abc.abstractmethod
    def prob_user_given_item(self, item, user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        pass
    
    @abc.abstractmethod
    def prob_item(self, item):
        '''Probability of seeing a given item. $P(i)$'''
        pass
    
    @abc.abstractmethod
    def num_items(self):
        '''Number of items'''
        pass

    @abc.abstractmethod
    def num_tags(self):
        '''Number of tags'''
        pass
    
    @abc.abstractmethod
    def num_users(self):
        '''Number of users'''
        pass
    
    @abc.abstractmethod
    def num_annotations(self):
        '''Number of annotations'''
        pass
    
    #Log methods are useful in case of underflows
    def log_prob_tag(self, tag):
        '''Log probability of seeing a given tag. $P(t)$'''
        return np.log2(self.prob_tag(tag))
    
    def log_prob_tag_given_item(self, item, tag):
        '''Log probability of seeing a given tag for an item. $P(t|i)$'''
        return np.log2(self.prob_tag_given_item(item, tag))
    
    def log_prob_user(self, user):
        '''Log probability of seeing an user. $P(u)$'''
        return np.log2(self.prob_user(user))
    
    def log_prob_user_given_item(self, item, user):
        '''Log probability of seeing an user given an item. $P(u|i)$'''
        return np.log2(self.prob_user_given_item(item, user))
    
    def log_prob_item(self, item):
        '''Log probability of seeing a given item. $P(i)$'''
        return np.log2(self.prob_item(item))

    #Vector methods
    def vect_prob_item(self, items):
        '''Computers the P(i) for a vector of items'''
        return self._vect_prob_item(self, items)
    
    def vect_prob_tag(self, tags):
        '''Computers the P(t) for a vector of tags'''
        return self._vect_prob_tag(self, tags)
        
    def vect_prob_user(self, users):
        '''Computers the P(u) for a vector of users'''
        return self._vect_prob_user(self, users)
    
    def vect_prob_tag_given_item(self, items, tag):
        '''Computers the P(t|i) for a vector of items'''
        return self._vect_prob_tag_given_item(self, items, tag)
        
    def vect_prob_user_given_item(self, items, user):
        '''Computers the P(u|i) for a vector of items'''
        return self._vect_prob_user_given_item(self, items, user)
    
    def vect_log_prob_item(self, items):
        '''Computers the log P(i) for a vector of items'''
        return self._vect_log_prob_item(self, items)
        
    def vect_log_prob_tag(self, tags):
        '''Computers the log P(t) for a vector of tags'''
        return self._vect_log_prob_tag(self, tags)
    
    def vect_log_prob_user(self, users):
        '''Computers the log P(u) for a vector of users'''
        return self._vect_log_prob_user(self, users)
    
    def vect_log_prob_tag_given_item(self, items, tag):
        '''Computers the log P(t|i) for a vector of items'''
        return self._vect_log_prob_tag_given_item(self, items, tag)
        
    def vect_log_prob_user_given_item(self, items, user):
        '''Computers the log P(u|i) for a vector of items'''
        return self._vect_log_prob_user_given_item(self, items, user)
    
    #Ugly hack, see if we can do better later.
    #It is ugly because it needs to be redone for every overwritten method.
    _vect_prob_item = np.vectorize(prob_item)
    _vect_prob_tag  = np.vectorize(prob_tag)
    _vect_prob_user = np.vectorize(prob_user)
    _vect_prob_tag_given_item  = np.vectorize(prob_tag_given_item)
    _vect_prob_user_given_item = np.vectorize(prob_user_given_item)
    
    _vect_log_prob_item = np.vectorize(log_prob_item)
    _vect_log_prob_tag  = np.vectorize(log_prob_tag)
    _vect_log_prob_user = np.vectorize(log_prob_user)
    _vect_log_prob_tag_given_item  = np.vectorize(log_prob_tag_given_item)
    _vect_log_prob_user_given_item = np.vectorize(log_prob_user_given_item)