# -*- coding: utf8
'''
Classes which computes item recommendations for a given user.
'''
from __future__ import division, print_function

import abc

class Recommender(object):
    '''Base Recommender, defines the relevant_items method'''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def relevance(self, user, item):
        '''Returns a the relevance of an item to a user'''
        pass

class ProbabilityReccomender(Recommender):
    '''
    Computes relevant items based on probability estimates.
    '''
    
    def __init__(self, estimator):
        super(ProbabilityReccomender, self).__init__()
        self.estimator = estimator
    
    def relevance(self, user, item):
        relevance = \
            self.estimator.log_prob_user_given_item(item, user) + \
            self.estimator.log_prob_item(item)
            
        return relevance