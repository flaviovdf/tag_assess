# -*- coding: utf8
'''Estimator based on smoothing methods'''

from __future__ import division, print_function

from collections import defaultdict
from tagassess.probability_estimates import ProbabilityEstimator

import numexpr as ne
import numpy as np

class SmoothEstimator(ProbabilityEstimator):
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
    def __init__(self, smooth_func, lambda_, annotation_it, cache = True):
        super(SmoothEstimator, self).__init__()
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
        self.user_tags = {}
        
        self.cache = cache
        if self.cache:
            self.pti_cache = {}
        else:
            self.pti_cache = None
        
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
            tag = annotation['tag']
            item = annotation['item']
            user = annotation['user']
            
            tag_col_dict[tag] += 1
            item_col_dict[item] += 1
            item_tag_dict[item][tag] += 1
            
            #Updating user utags
            if user in self.user_tags:
                utags = self.user_tags[user]
            else:
                #If this becomes an overhead, change to set.
                utags = []
                self.user_tags[user] = utags
            
            if tag not in utags:
                utags.append(tag)

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
        p_tag_items = self.vect_prob_tag_given_item(items, tag)
        return ne.evaluate('sum(p_items * p_tag_items)')
    
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        key = (item, tag)
        if self.cache and key in self.pti_cache:
            return self.pti_cache[key]
        else:
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
                return_val = prob
            else:
                mle = self.tag_col_freq[tag] / self.n_annotations
                return_val = alpha * mle
            
            if self.cache:
                self.pti_cache[key] = return_val
            return return_val
    
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        if len(self.user_tags[user]) == 0:
            return 0
        else:
            atags = np.array(self.user_tags[user])
            prob_t = self.vect_prob_tag(atags)
            return prob_t.prod()
    
    def prob_user_given_item(self, item, user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        if len(self.user_tags[user]) == 0:
            return 0
        else:
            atags = np.array(self.user_tags[user])
            prob_ut = self.vect_prob_tag_given_item(item, atags)
            return prob_ut.prod()
    
    #Log methods
    def log_prob_user(self, user):
        '''
        Log probability of seeing an user. $P(u)$
        This method is useful when `prob_user` underflows.
        '''
        if len(self.user_tags[user]) == 0:
            return float('-inf')
        else:
            atags = np.array(self.user_tags[user])
            prob_t = self.vect_log_prob_tag(atags)
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
            prob_ut = self.vect_log_prob_tag_given_item(item, atags)
            return prob_ut.sum()
    
    #Vectorized methods
    _vect_prob_user = np.vectorize(prob_user)
    _vect_prob_item = np.vectorize(prob_item)
    _vect_prob_tag  = np.vectorize(prob_tag)
    
    _vect_prob_user_given_item = np.vectorize(prob_user_given_item)
    _vect_prob_tag_given_item  = np.vectorize(prob_tag_given_item)
    
    _vect_log_prob_user = np.vectorize(log_prob_user)
    _vect_log_prob_user_given_item = np.vectorize(log_prob_user_given_item)
    
    #Other methods
    def num_items(self):
        '''Number of items'''
        return len(self.item_col_mle)
    
    def num_tags(self):
        '''Number of tags'''
        return len(self.tag_col_freq)
    
    def num_users(self):
        '''Number of users'''
        return len(self.user_tags)
    
    def num_annotations(self):
        '''Number of annotations'''
        return self.n_annotations