# -*- coding: utf8
'''
Classes that compute:

    * P(i) = Probability of an item
    * P(t) = Probability of a tag
    * P(u) = Probability of an user
    * P(u|i) = Probability of an user given an item
    * P(t|i) = Probability of a tag given an item
'''
from __future__ import division, print_function

from collections import defaultdict

import abc
import numpy as np

def assert_good_prob(func):
    '''Utility method to check probabilities'''
    
    def check(*args, **kwargs): 
        '''decorator'''
        prob = func(*args, **kwargs)
        if not prob <= 1 + 10e-14:
            raise AssertionError('Invalid prob. = %.5f %s'%(prob))
        return prob
    return check

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

class MLE(ProbabilityEstimator):
    '''Estimations completely based on maximum likelihood'''

    def __init__(self):
        super(MLE, self).__init__()
        self.n_annotations = 0
        self.item_tag_freq = defaultdict(lambda: defaultdict(int))
        self.tag_col_freq = defaultdict(int)
        self.item_col_freq = defaultdict(int)
        self.item_user_freq = defaultdict(lambda: defaultdict(int))
        self.user_tag_freq = defaultdict(lambda: defaultdict(int))
        self.user_col_freq = defaultdict(int)
        
    def open(self, annotation_it):
        '''
        Computes initial indexes based on the iterator
        
        Arguments
        ---------
        annotation_it: iterable
            An iterable with annotations
        '''
        
        #For this class we need user and item indexes
        self.n_annotations = 0
        for annotation in annotation_it:
            self.n_annotations += 1
            
            tag = annotation.get_tag()
            item = annotation.get_item()
            user = annotation.get_user()
            
            self.item_tag_freq[item][tag] += 1
            self.tag_col_freq[tag] += 1
            self.item_col_freq[item] += 1
            self.item_user_freq[item][user] += 1
            self.user_tag_freq[user][tag] += 1
            self.user_col_freq[user] += 1
            
    @assert_good_prob
    def prob_tag(self, tag):
        '''Probability of seeing a given tag. $P(t)$'''
        return self.tag_col_freq[tag] / self.n_annotations
    
    @assert_good_prob
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        sum_local = sum(self.item_tag_freq[item].values())
        return self.item_tag_freq[item][tag] / sum_local
    
    @assert_good_prob
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        return self.user_col_freq[user] / self.n_annotations
    
    @assert_good_prob
    def prob_user_given_item(self, item, user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        sum_local = sum(self.item_user_freq[item].values())
        return self.item_user_freq[item][user] / sum_local
    
    @assert_good_prob
    def prob_item(self, item):
        '''Probability of seeing a given item. $P(i)$'''
        return self.item_col_freq[item] / self.n_annotations

class SmoothedItems(MLE):
    '''
    In this approach, items are smoothed according to
    a smoothing function which will consider global information.
    
    In details:
        * $P(t|i) = P(t|M_i)$ where, $M_i$ is a smoothed model of the items
    '''
    
    def __init__(self, smooth_func, lambda_):
        super(SmoothedItems, self).__init__()
        self.smooth_func = smooth_func
        self.lambda_ = lambda_
    
    def __estimator(self, item, tag):
        '''Calls the smoothing method accordingly'''
        sum_local = sum(self.item_tag_freq[item].values())
        prob, alpha = self.smooth_func(self.item_tag_freq[item][tag],
                                        sum_local,
                                        self.tag_col_freq[tag],
                                        self.n_annotations,
                                        self.lambda_)
        return prob, alpha
    
    @assert_good_prob
    def prob_tag(self, tag):
        '''Probability of seeing a given tag. $P(t)$'''
        sum_each_item = 0
        for item in self.item_col_freq.keys():
            prob_tag_given_item = self.prob_tag_given_item(item, tag)
            prob_item = self.prob_item(item)
            sum_each_item += prob_tag_given_item * prob_item
        return sum_each_item
    
    @assert_good_prob
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        prob, alpha = self.__estimator(item, tag)
        
        if self.item_tag_freq[item][tag] != 0:
            return prob    
        else:
            mle_tag = super(SmoothedItems, self).prob_tag(tag)
            return alpha * mle_tag

class SmoothedItemsUsersAsTags(SmoothedItems):
    '''
    Implementation of the approach proposed in:
    
    Personalization of Tagging Systems, 
    Wang, Jun, Clements Maarten, Yang J., de Vries Arjen P., and Reinders Marcel J. T. , 
    Information Processing and Management, Volume 46, Issue 1, p.58-70, (2010)
    
    In details:
        * $P(t|i) = P(t|M_i)$ where, $M_i$ is a smoothed model of the items
        * $P(u)$ and $P(u|i)$ considers users as tags. More specifically, the past
          tags used by the user. So, these two functions will make use of $P(t)$ and $P(t|i)$.
    '''
    
    def __init__(self, smooth_func, lambda_):
        super(SmoothedItemsUsersAsTags, self).__init__(smooth_func, lambda_)
        self.smooth_func = smooth_func
        self.lambda_ = lambda_
    
    @assert_good_prob
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        if len(self.user_tag_freq[user]) == 0:
            return 0
        else:
            product = 1
            for tag in self.user_tag_freq[user]:
                product *= self.prob_tag(tag)
            return product
    
    @assert_good_prob
    def prob_user_given_item(self, item, user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        if len(self.user_tag_freq[user]) == 0:
            return 0
        else:
            product = 1
            for tag in self.user_tag_freq[user]:
                product *= self.prob_tag_given_item(item, tag)
            return product
    
    def log_prob_user(self, user):
        '''
        Log probability of seeing an user. $P(u)$
        This method is useful when `prob_user` underflows.
        '''
        if len(self.user_tag_freq[user]) == 0:
            return float('-inf')
        else:
            add = 0
            for tag in self.user_tag_freq[user]:
                add += self.log_prob_tag(tag)
            return add
    
    def log_prob_user_given_item(self, item, user):
        '''
        Log probability of seeing an user given an item. $P(u|i)$.
        This method is useful when `prob_user_given_item` underflows.
        '''
        if len(self.user_tag_freq[user]) == 0:
            return float('-inf')
        else:
            add = 0
            for tag in self.user_tag_freq[user]:
                add += self.log_prob_tag_given_item(item, tag)
            return add