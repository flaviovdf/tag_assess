# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from tagassess import entropy

import numpy as np

class ValueCalculator(object):
    '''
    Class used to compute values. 
    Contains basic value functions and filtering.
    '''
    
    def __init__(self, estimator, recommender):
        self.est = estimator
        self.recc = recommender
        
    def item_value(self, user):
        '''
        Creates an array for the relevance of each item to the given user.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        return_val = np.zeros(self.est.num_items())
        for item in xrange(len(return_val)):
            relevance = self.recc.relevance(user, item)
            return_val[item] = relevance
        return return_val
    
    def tag_value_personalized(self, user, gamma_items = None):
        '''
        Creates an array for the value of each tag to the given user.
        In details, this computes:
        
        D( P(i | t, u) || P(i | u) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        return_val = np.zeros(self.est.num_tags())
        for tag in xrange(len(return_val)):
            vp_iu = self.rnorm_prob_items_given_user(user, gamma_items)
            vp_itu = self.rnorm_prob_items_given_user_tag(user, tag, 
                                                          gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            return_val[tag] = tag_val
        return return_val
    
    def tag_value_item_search(self, gamma_items = None):
        '''
        Creates an array for the value of each tag in a global context.
        
        In details, this computes:
        
        D( P(i | t) || P(i) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        vp_i = self.rnorm_prob_items(gamma_items)
        
        return_val = np.zeros(self.est.num_tags())
        for tag in xrange(len(return_val)):
            vp_it = self.rnorm_prob_items_given_tag(tag, gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_it, vp_i)
            return_val[tag] = tag_val
        return return_val

    def tag_value_per_user_search(self, user, gamma_items = None):
        '''
        Creates an array for the value of each tag in a global context.
        
        In details, this computes:
        
        D( P(i | t) || P(i | u) ),
        
        where D is the kullback-leiber divergence.
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        
        return_val = np.zeros(self.est.num_items())
        for tag in xrange(len(return_val)):
            vp_iu = self.rnorm_prob_items_given_user(user, gamma_items)
            vp_it = self.rnorm_prob_items_given_tag(tag, gamma_items)
            
            tag_val = entropy.kullback_leiber_divergence(vp_it, vp_iu)
            return_val[tag] = tag_val
        return return_val

    def rnorm_prob_items_given_user(self, user, gamma_items = None):
        '''
        Computes P(I|u)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if gamma_items is None:
            items = range(self.est.num_items())
        else:
            items = gamma_items
        
        p_u = self.est.prob_user(user)
        vp_i = self.est.vect_prob_item(items)
        vp_ui = self.est.vect_prob_user_given_item(items, user)
        
        vp_iu = vp_ui * (vp_i / p_u)
        vp_iu = vp_iu / vp_iu.sum()
        return vp_iu

    def rnorm_prob_items_given_user_tag(self, user, tag, gamma_items = None):
        '''
        Computes P(I|u,t)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if gamma_items is None:
            items = range(self.est.num_items())
        else:
            items = gamma_items
        
        p_t = self.est.prob_tag(tag)
        p_u = self.est.prob_user(user)
        
        vp_i = self.est.vect_prob_item(items)
        vp_ui = self.est.vect_prob_user_given_item(items, user)
        vp_ti = self.est.vect_prob_tag_given_item(items, tag)

        vp_itu = vp_ti * vp_ui * (vp_i / (p_u * p_t))
        vp_itu = vp_itu / vp_itu.sum()
        return vp_itu
    
    def rnorm_prob_items_given_tag(self, tag, gamma_items = None):
        '''
        Computes P(I|t)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if gamma_items is None:
            items = range(self.est.num_items())
        else:
            items = gamma_items
        
        p_t = self.est.prob_tag(tag)
        vp_i = self.est.vect_prob_item(items)
        vp_ti = self.est.vect_prob_tag_given_item(items, tag)
        
        vp_it = vp_ti * (vp_i / p_t)
        vp_it = vp_it / vp_it.sum()
        return vp_it
    
    def rnorm_prob_items(self, gamma_items = None):
        '''
        Computes P(I)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if gamma_items is None:
            items = range(self.est.num_items())
        else:
            items = gamma_items
            
        vp_i = self.est.vect_prob_item(items)
        vp_i = vp_i / vp_i.sum()
        return vp_i