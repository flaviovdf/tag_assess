# -*- coding: utf8
'''
Estimator based on random walks
'''
from __future__ import division, print_function

from collections import defaultdict
from tagassess import graph
from tagassess.probability_estimates import ProbabilityEstimator

import networkx as nx
import numpy as np

class RWEstimator(ProbabilityEstimator):
    '''
    Implementation of the probability estimators based random walks in tag x
    item graph. For more information check:
    
    Algorithms for Estimating Relative Importance in Networks
    Scott White, Padhraic Smyth
    SIGKDD 2003
    
    In details:
        * $P(t)$ is the page rank value of the tag
        * $P(i)$ is also the page rank value
        * $P(t|i)$ is based on the personalized page rank vector centering on the tag. That is:
          $P(t|i) = P(i|t) * P(i) / P(t)$, where $P(i|t)$ is the personalized page rank on $t$.
        * $P(u)$ and $P(u|i)$ considers users as tags. More specifically, the past
          tags used by the user. So, these two functions will make use of $P(t)$ and $P(t|i)$.
    '''
    def __init__(self, annotation_it, cache = True):
        super(RWEstimator, self).__init__()
        
        self.cache = cache
        if self.cache:
            self.pti_cache = {}
        else:
            self.pti_cache = None
        
        self.n_annotations = 0
        self.n_items = 0
        self.n_users = 0
        self.n_tags = 0
        
        self.user_tags = {}
        self.di_graph = None
        self.item_probs = None
        self.tag_probs = None
        self.pers_dict = None
        
        self.__populate(annotation_it)

    def __populate(self, annotation_it):
        '''
        Computes initial indexes based on the the graph
        
        Arguments
        ---------
        annotation_it: iterable
            An iterable with annotations
        '''
        tag_to_item = defaultdict(set)
        item_to_tag = defaultdict(set)
        
        for annotation in annotation_it:
            self.n_annotations += 1
            user = annotation['user']
            item = annotation['item']
            tag = annotation['tag']
            
            tag_to_item[tag].add(item)
            item_to_tag[item].add(tag)
            
            #Updating user user tags
            if user in self.user_tags:
                utags = self.user_tags[user]
            else:
                utags = []
                self.user_tags[user] = utags
            
            if tag not in utags:
                utags.append(tag)
        
        #Initialize variables
        num_tags, num_items, edge_gen = graph.iedge_from_indexes(tag_to_item,
                                                                 item_to_tag)
        self.di_graph = graph.create_nxgraph(edge_gen)
        self.n_items = num_items
        self.n_tags  = num_tags
        self.n_users = len(self.user_tags)
        
        #Compute pagerank which leads to tag and item probs
        page_ranks_dict = nx.pagerank_scipy(self.di_graph)
        
        self.item_probs = np.zeros(shape=self.n_items)
        self.tag_probs = np.zeros(shape=self.n_tags)
        
        self.pers_dict = {}
        for node_id, page_rank in page_ranks_dict.iteritems():
            self.pers_dict[node_id] = 0
            if node_id < self.n_tags:
                self.tag_probs[node_id] = page_rank
            else:
                item_id = node_id - self.n_tags
                self.item_probs[item_id] = page_rank

        #Renormalization
        self.item_probs /= self.item_probs.sum()
        self.tag_probs  /= self.tag_probs.sum()
        
    def prob_item(self, item):
        '''Probability of seeing a given item. $P(i)$'''
        return self.item_probs[item]
    
    def prob_tag(self, tag):
        '''Probability of seeing a given tag. $P(t)$'''
        return self.tag_probs[tag]
    
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        key = (item, tag)
        if self.cache and key in self.pti_cache:
            return self.pti_cache[key]
        else:
            #Personalized page rank
            self.pers_dict[tag] = 1
            pager = nx.pagerank_scipy(self.di_graph,
                                         personalization = self.pers_dict)
            self.pers_dict[tag] = 0
            
            #Doing P(t|i) = P(i|t) * P(t) / P(i)
            n_nodes = self.n_tags + self.n_items
            prob_it = np.array([pager[i] for i in xrange(self.n_tags, n_nodes)])
            prob_it /= prob_it.sum()
            
            prob_t = self.prob_tag(tag)
            prob_i = self.prob_item(item)
            
            factor = prob_t / prob_i
            prob_ti = prob_it * factor
            
            #Cache
            return_val = prob_ti[item]
            if self.cache:
                for item_id in xrange(self.n_items):
                    prob = prob_ti[item_id]
                    self.pti_cache[(item_id, tag)] = prob
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