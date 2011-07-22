# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from tagassess import entropy

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
        Creates a generator for the relevance of each item to the given user.
        The generator will yield the tuple: (item_relevance, item).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        return_val = {}
        for item in xrange(self.est.num_items()):
            relevance = self.recc.relevance(user, item)
            return_val[item] = relevance
        return return_val
    
    def tag_value_ucontext(self, user, gamma_items = None):
        '''
        Creates a generator for the value of each tag to the given user.
        The generator will yield the tuple: (tag_value, tag).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if gamma_items is not None:
            items = gamma_items
        else:
            items = range(self.est.num_items())
                         
        est = self.est
        p_i = est.vect_prob_item(items)
        p_ui = est.vect_prob_user_given_item(items, user)
        p_u = est.prob_user(user) 
        
        return_val = {}
        for tag in xrange(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                return_val[tag] = 0
            
            p_ti = est.vect_prob_tag_given_item(items, tag)
            
            p_iu = p_ui * p_i / p_u
            p_itu = p_ti * p_ui * p_i / (p_u * p_t)
            
            #Renormalization is necessary
            p_iu /= p_iu.sum()
            p_itu /= p_itu.sum() 
            
            tag_val = entropy.kullback_leiber_divergence(p_itu, p_iu)
            return_val[tag] = tag_val
        return return_val
    
    def tag_value_gcontext(self, gamma_items = None):
        '''
        Creates a generator for the value of each tag in a global context.
        The generator will yield the tuple: (tag_value, tag).
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if gamma_items is not None:
            items = gamma_items
        else:
            items = range(self.est.num_items())
            
        est = self.est
        p_i = est.vect_prob_item(items)
        
        return_val = {}    
        for tag in xrange(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                return_val[tag] = 0
            
            p_ti = est.vect_prob_tag_given_item(items, tag)
            p_it = p_ti * p_i / p_t
            
            #Renormalization is necessary
            p_it /= p_it.sum()
            p_i /= p_i.sum()
            
            tag_val = entropy.kullback_leiber_divergence(p_it, p_i)
            return_val[tag] = tag_val
        return return_val