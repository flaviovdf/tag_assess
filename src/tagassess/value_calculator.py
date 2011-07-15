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
        
    def iitem_value(self, user, items_to_compute=None):
        '''
        Creates a generator for the relevance of each item to the given user.
        The generator will yield the tuple: (item_relevance, item).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if items_to_compute:
            items = items_to_compute
        else:
            items = range(self.est.num_items())
    
        for item in items:
            relevance = self.recc.relevance(user, item)
            yield relevance, item
    
    def itag_value_ucontext(self, user, items_to_compute=None,
                            tags_to_consider=None):
        '''
        Creates a generator for the value of each tag to the given user.
        The generator will yield the tuple: (tag_value, tag).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if items_to_compute:
            items = items_to_compute
        else:
            items = range(self.est.num_items())
                         
        est = self.est
        p_i = est.vect_prob_item(items)
        p_ui = est.vect_prob_user_given_item(items, user)
        p_u = est.prob_user(user) 
        
        if tags_to_consider:
            tags = tags_to_consider
        else:
            tags = range(self.est.num_tags())
            
        for tag in tags:
            p_ti = est.vect_prob_tag_given_item(items, tag)
            p_t = est.prob_tag(tag)
            
            p_iu = p_ui * p_i / p_u
            p_itu = p_ti * p_ui * p_i / (p_u * p_t)
            
            #Renormalization is necessary
            p_iu /= p_iu.sum()
            p_itu /= p_itu.sum() 
            
            tag_val = entropy.kullback_leiber_divergence(p_itu, p_iu)
            yield tag_val, tag
    
    def itag_value_gcontext(self, items_to_compute=None, 
                            tags_to_consider=None):
        '''
        Creates a generator for the value of each tag in a global context.
        The generator will yield the tuple: (tag_value, tag).
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if items_to_compute:
            items = items_to_compute
        else:
            items = range(self.est.num_items())
                         
        est = self.est
        p_i = est.vect_prob_item(items)
        
        if tags_to_consider:
            tags = tags_to_consider
        else:
            tags = range(self.est.num_tags())
            
        for tag in tags:
            p_ti = est.vect_prob_tag_given_item(items, tag)
            p_t = est.prob_tag(tag)
            p_it = p_ti * p_i / p_t
            
            #Renormalization is necessary
            p_it /= p_it.sum()
            p_i /= p_i.sum()
            
            tag_val = entropy.kullback_leiber_divergence(p_it, p_i)
            yield tag_val, tag