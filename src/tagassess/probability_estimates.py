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
import numexpr as ne
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

    #Vectorized methods
    #TODO: Ugly hack, see if we can do better later.
    #It is ugly because it needs to be redone for every overwritten method.
    #We could use reflection, but lets keep this for now. Small module.
    vect_prob_item = np.vectorize(prob_item)
    vect_prob_tag  = np.vectorize(prob_tag)
    vect_prob_user = np.vectorize(prob_user)
    vect_prob_tag_given_item  = np.vectorize(prob_tag_given_item)
    vect_prob_user_given_item = np.vectorize(prob_user_given_item)
    
    vect_log_prob_item = np.vectorize(log_prob_item)
    vect_log_prob_tag  = np.vectorize(log_prob_tag)
    vect_log_prob_user = np.vectorize(log_prob_user)
    vect_log_prob_tag_given_item  = np.vectorize(log_prob_tag_given_item)
    vect_log_prob_user_given_item = np.vectorize(log_prob_user_given_item)

class MLE(ProbabilityEstimator):
    '''Estimations completely based on maximum likelihood'''

    def __init__(self):
        super(MLE, self).__init__()
        self.n_annotations = 0
        
        self.item_tag_freq = {}
        self.item_user_freq = {}
        self.item_tag_freq_ids = defaultdict(lambda: defaultdict(lambda: -1))
        self.item_user_freq_ids = defaultdict(lambda: defaultdict(lambda: -1))
        
        self.tag_col_freq = None
        self.item_col_freq = None
        self.user_col_freq = None
        
    def open(self, annotation_it):
        '''
        Computes initial indexes based on the iterator
        
        Arguments
        ---------
        annotation_it: iterable
            An iterable with annotations
        '''
        item_tag_dict = defaultdict(lambda: defaultdict(int))
        item_user_dict = defaultdict(lambda: defaultdict(int))
        
        tag_col_dict = defaultdict(int)
        item_col_dict = defaultdict(int)
        user_col_dict = defaultdict(int)
        
        max_tag = 0
        max_user = 0
        max_item = 0
        
        #For this class we need user and item indexes
        self.n_annotations = 0
        for annotation in annotation_it:
            self.n_annotations += 1
            
            tag = annotation.get_tag()
            item = annotation.get_item()
            user = annotation.get_user()
            
            item_tag_dict[item][tag] += 1
            item_user_dict[item][user] += 1
            
            tag_col_dict[tag] += 1
            item_col_dict[item] += 1
            user_col_dict[user] += 1
            
            if user > max_user:
                max_user = user
                
            if tag > max_tag:
                max_tag = tag
                
            if item > max_item:
                max_item = item
            
            self._extra_updates(user, tag, item)
        
        self.tag_col_freq = np.zeros(shape = (max_tag + 1,))
        for tag in tag_col_dict:
            self.tag_col_freq[tag] = tag_col_dict[tag]
        
        self.user_col_freq = np.zeros(shape = (max_user + 1,))
        for user in user_col_dict:
            self.user_col_freq[user] = user_col_dict[user]
        
        self.item_col_freq = np.zeros(shape = (max_item + 1,))
        for item in item_col_dict:
            self.item_col_freq[item] = item_col_dict[item]
            
            shape = (len(item_tag_dict[item]),)
            self.item_tag_freq[item] = np.ndarray(shape=shape)
            for i, tag in enumerate(item_tag_dict[item]):
                self.item_tag_freq[item][i] = item_tag_dict[item][tag]
                self.item_tag_freq_ids[item][tag] = i
                
            shape = (len(item_user_dict[item]),)
            self.item_user_freq[item] = np.ndarray(shape=shape)
            for i, user in enumerate(item_user_dict[item]):
                self.item_user_freq[item][i] = item_user_dict[item][user]
                self.item_user_freq_ids[item][user] = i
    
    def _extra_updates(self, user, tag, item):
        '''
        Performs any extra update on the open method. This
        method should be overridden by subclasses
        '''
        pass
            
    @assert_good_prob
    def prob_tag(self, tag):
        '''Probability of seeing a given tag. $P(t)$'''
        return self.tag_col_freq[tag] / self.n_annotations
    
    @assert_good_prob
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        sum_local = self.item_tag_freq[item].sum()
        tid = self.item_tag_freq_ids[item][tag]
        
        if tid == -1:
            return 0
        else:
            return self.item_tag_freq[item][tid] / sum_local
    
    @assert_good_prob
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        return self.user_col_freq[user] / self.n_annotations
    
    @assert_good_prob
    def prob_user_given_item(self, item, user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        sum_local = self.item_user_freq[item].sum()
        uid = self.item_user_freq_ids[item][user]
        
        if uid == -1:
            return 0
        else:
            return self.item_user_freq[item][uid] / sum_local
    
    @assert_good_prob
    def prob_item(self, item):
        '''Probability of seeing a given item. $P(i)$'''
        return self.item_col_freq[item] / self.n_annotations

    #Vectorized methods
    vect_prob_item = np.vectorize(prob_item)
    vect_prob_tag  = np.vectorize(prob_tag)
    vect_prob_user = np.vectorize(prob_user)
    vect_prob_tag_given_item  = np.vectorize(prob_tag_given_item)
    vect_prob_user_given_item = np.vectorize(prob_user_given_item)
    
    def valid_items(self):
        '''Items with non zero P(i)'''
        return self.item_col_freq.nonzero()[0]

    def valid_tags(self):
        '''Tags with non zero P(t)'''
        return self.tag_col_freq.nonzero()[0]
    
    def valid_users(self):
        '''Users with non zero P(u)'''
        return self.user_col_freq.nonzero()[0]
    
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
    
    def open(self, annotation_it):
        super(SmoothedItems, self).open(annotation_it)
        del self.item_user_freq
        del self.user_col_freq
    
    @assert_good_prob
    def prob_tag(self, tag):
        '''Probability of seeing a given tag. $P(t)$'''
        items = np.arange(len(self.item_col_freq))
        p_items = self.vect_prob_item(self, items)
        p_tag_items = self.vect_prob_tag_given_item(self, items, tag)
        return ne.evaluate('sum(p_items * p_tag_items)')
    
    @assert_good_prob
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        sum_local = self.item_tag_freq[item].sum()
        tid = self.item_tag_freq_ids[item][tag]
        
        if tid == -1:
            alpha = self.smooth_func(0,
                                     sum_local,
                                     self.tag_col_freq[tag],
                                     self.n_annotations,
                                     self.lambda_)[1]
            mle_tag = super(SmoothedItems, self).prob_tag(tag)
            return alpha * mle_tag
        else:
            prob = self.smooth_func(self.item_tag_freq[item][tid],
                                    sum_local,
                                    self.tag_col_freq[tag],
                                    self.n_annotations,
                                    self.lambda_)[0]
            return prob

    #Vectorized methods
    vect_prob_tag  = np.vectorize(prob_tag)
    vect_prob_tag_given_item  = np.vectorize(prob_tag_given_item)

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
        self.user_tags = {}
    
    def _extra_updates(self, user, tag, item):
        if user in self.user_tags:
            tags = self.user_tags[user]
        else:
            #If this becomes an overhead, change to set.
            tags = []
            self.user_tags[user] = tags
        
        if tag not in tags:
            tags.append(tag)
    
    @assert_good_prob
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        if len(self.user_tags[user]) == 0:
            return 0
        else:
            atags = np.array(self.user_tags[user])
            prob_t = self.vect_prob_tag(self, atags)
            return prob_t.prod()
    
    @assert_good_prob
    def prob_user_given_item(self, item, user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        if len(self.user_tags[user]) == 0:
            return 0
        else:
            atags = np.array(self.user_tags[user])
            prob_ut = self.vect_prob_tag_given_item(self, item, atags)
            return prob_ut.prod()
    
    def log_prob_user(self, user):
        '''
        Log probability of seeing an user. $P(u)$
        This method is useful when `prob_user` underflows.
        '''
        if len(self.user_tags[user]) == 0:
            return float('-inf')
        else:
            atags = np.array(self.user_tags[user])
            prob_t = self.vect_log_prob_tag(self, atags)
            return prob_t.sum()
    
    def log_prob_user_given_item(self, item, user):
        '''
        Log probability of seeing an user given an item. $P(u|i)$.
        This method is useful when `prob_user_given_item` underflows.
        '''
        if len(self.user_tags[user]) == 0:
            return float('-inf')
        else:
            atags = np.array(self.user_tags[user])
            prob_ut = self.vect_log_prob_tag_given_item(self, item, atags)
            return prob_ut.sum()
    
    #Vectorized methods
    vect_prob_user = np.vectorize(prob_user)
    vect_prob_user_given_item = np.vectorize(prob_user_given_item)
    
    vect_log_prob_user = np.vectorize(log_prob_user)
    vect_log_prob_user_given_item = np.vectorize(log_prob_user_given_item)