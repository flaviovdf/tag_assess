# -*- coding: utf8
'''
Classes which computes item recommendations for a given user.
'''
from __future__ import division, print_function

import abc
import numpy as np

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
            np.log2(self.estimator.prob_user_given_item(item, user)) + \
            np.log2(self.estimator.prob_item(item))
            
        return relevance