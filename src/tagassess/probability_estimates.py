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

from tagassess.dao.annotations import AnnotReader
from tagassess import index_creator

import abc

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
    def prob_user_given_item(self, user, tag):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        pass
    
    @abc.abstractmethod
    def prob_item(self, item):
        '''Probability of seeing a given item. $P(i)$'''
        pass

class MLE(ProbabilityEstimator):
    '''Estimations completely based on maximum likelihood'''

    def __init__(self, annotation_file, table):
        super(MLE, self).__init__()
        self.table = table
        self.annotation_reader = AnnotReader(annotation_file)

        self.n_annotations = 0
        self.item_tag_freq = {}
        self.tag_col_freq = {}
        self.item_col_freq = {}
        self.item_user_freq = {}
        self.user_col_freq = {}
        
    def open(self):
        '''Opens the annotation file and computes indexes'''
        self.annotation_reader.open_file()
        
        #For this class we need user and item indexes
        iterator = self.annotation_reader.iterate(self.table)
        self.item_tag_freq, self.item_col_freq, self.tag_col_freq = \
            index_creator.create_metrics_index(iterator, 'item', 'tag')

        iterator = self.annotation_reader.iterate(self.table)
        self.item_user_freq, aux, self.user_col_freq = \
            index_creator.create_metrics_index(iterator, 'item', 'user')
        del aux
    
        self.n_annotations = sum(self.tag_col_freq.values())
    
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
    
    def close(self):
        '''Closes the file'''
        self.annotation_reader.close_file()

class SmoothedItems(MLE):
    '''
    In this approach, items are smoothed according to
    a smoothing function which will consider global information.
    
    In details:
        * $P(t|i) = P(t|M_i)$ where, $M_i$ is a smoothed model of the items
    '''
    
    def __init__(self, annotation_file, table, smooth_func, lambda_):
        super(SmoothedItems, self).__init__(annotation_file, table)
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
    def prob_tag_given_item(self, item, tag):
        '''Probability of seeing a given tag for an item. $P(t|i)$'''
        prob, alpha = self.__estimator(item, tag)
        
        if self.item_tag_freq[item][tag] != 0:
            return prob    
        else:
            return alpha * self.prob_tag(tag)

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
    
    def __init__(self, annotation_file, table, smooth_func, lambda_):
        super(SmoothedItemsUsersAsTags, self).__init__(annotation_file, table, 
                                                       smooth_func, lambda_)
        self.smooth_func = smooth_func
        self.lambda_ = lambda_
    
    def _get_user_tags(self, user):
        '''Returns the tags used by the given user'''
        iterator = self.annotation_reader.iterate(self.table,
                                                  'USER == ' + str(user))
        return set(a.get_tag() for a in iterator)
    
    @assert_good_prob
    def prob_user(self, user):
        '''Probability of seeing an user. $P(u)$'''
        user_tags = self._get_user_tags(user)
        if len(user_tags) == 0:
            return 0
        else:
            product = 1
            for tag in user_tags:
                product *= self.prob_tag(tag)
            return product
    
    @assert_good_prob
    def prob_user_given_item(self, item, user):
        '''Probability of seeing an user given an item. $P(u|i)$'''
        user_tags = self._get_user_tags(user)
        if len(user_tags) == 0:
            return 0
        else:
            product = 1
            for tag in user_tags:
                product *= self.prob_tag_given_item(item, tag)
            return product