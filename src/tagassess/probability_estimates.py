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

    #Vector methods
    #Ugly hack, see if we can do better later.
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

class SmoothedItemsUsersAsTags(ProbabilityEstimator):
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
    def __init__(self, smooth_func, lambda_, annotation_it):
        super(SmoothedItemsUsersAsTags, self).__init__()
        self.n_annotations = 0
        self.smooth_func = smooth_func
        self.lambda_ = lambda_
        
        #These will be numpy arrays
        self.item_col_mle = None
        self.tag_col_freq = None
        self.item_local_sums = None
        
        #I prefer not to use defaultdicts here.
        #Key error are better than wrong values.
        self.item_tag_freq = {}
        self.pti_cache = {}
        self.user_tags = {}
        
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
        
        max_tag = 0
        max_item = 0
        
        #For this class we need user and item indexes
        self.n_annotations = 0
        for annotation in annotation_it:
            self.n_annotations += 1
            
            #Initial updates
            tag = annotation.get_tag()
            item = annotation.get_item()
            user = annotation.get_user()
            
            tag_col_dict[tag] += 1
            item_col_dict[item] += 1
            item_tag_dict[item][tag] += 1
            
            #Updating user tags
            if user in self.user_tags:
                tags = self.user_tags[user]
            else:
                #If this becomes an overhead, change to set.
                tags = []
                self.user_tags[user] = tags
            
            if tag not in tags:
                tags.append(tag)

            #Tag, item and user id spaced being defined            
            if tag > max_tag:
                max_tag = tag
                
            if item > max_item:
                max_item = item
        
        self.tag_col_freq = np.zeros(shape = max_tag + 1)
        for tag in tag_col_dict:
            self.tag_col_freq[tag] = tag_col_dict[tag]
        
        self.item_col_mle = np.zeros(shape = max_item + 1)
        self.item_local_sums = np.zeros(shape = max_item + 1)
        for item in item_col_dict:
            self.item_col_mle[item] = item_col_dict[item] / self.n_annotations
            
            sum_local = sum(item_tag_dict[item].values())
            self.item_local_sums[item] = sum_local
            
            for tag in item_tag_dict[item]:
                self.item_tag_freq[item, tag] = item_tag_dict[item][tag]
                
    def prob_item(self, item):
        '''Probability of seeing a given item. $P(i)$'''
        return self.item_col_mle[item]
    
    def prob_tag(self, tag):
        '''Probability of seeing a given tag. $P(t)$'''
        items = np.arange(len(self.item_col_mle))
        p_items = self.item_col_mle
        p_tag_items = self.vect_prob_tag_given_item(self, items, tag)
        return ne.evaluate('sum(p_items * p_tag_items)')
    
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        key = (item, tag)
        if (item, tag) not in self.pti_cache:
            if key in self.item_tag_freq:
                local_freq = self.item_tag_freq[key]
            else:
                local_freq = 0
            
            sum_local = self.item_local_sums[item]
            prob, alpha = self.smooth_func(local_freq,
                                           sum_local,
                                           self.tag_col_freq[tag],
                                           self.n_annotations,
                                           self.lambda_)
            
            if local_freq:
                self.pti_cache[item, tag] = prob
            else:
                mle = self.tag_col_freq[tag] / self.n_annotations
                self.pti_cache[item, tag] = alpha * mle
                
        return self.pti_cache[item, tag]
    
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        if len(self.user_tags[user]) == 0:
            return 0
        else:
            atags = np.array(self.user_tags[user])
            prob_t = self.vect_prob_tag(self, atags)
            return prob_t.prod()
    
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
    vect_prob_item = np.vectorize(prob_item)
    vect_prob_tag  = np.vectorize(prob_tag)
    
    vect_prob_user_given_item = np.vectorize(prob_user_given_item)
    vect_prob_tag_given_item  = np.vectorize(prob_tag_given_item)
    
    vect_log_prob_user = np.vectorize(log_prob_user)
    vect_log_prob_user_given_item = np.vectorize(log_prob_user_given_item)
    
    def valid_items(self):
        '''Items with non zero P(i)'''
        return self.item_col_mle.nonzero()[0]

    def valid_tags(self):
        '''Tags with non zero P(t)'''
        return self.tag_col_freq.nonzero()[0]