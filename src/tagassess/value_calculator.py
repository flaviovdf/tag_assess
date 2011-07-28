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
        Creates a map for the relevance of each item to the given user.
        
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
        Creates a map for the value of each tag to the given user.
        
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
        vp_i = est.vect_prob_item(items)
        vp_ui = est.vect_prob_user_given_item(items, user)
        p_u = est.prob_user(user) 
        
        return_val = {}
        for tag in xrange(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                return_val[tag] = 0
                continue
            
            vp_ti = est.vect_prob_tag_given_item(items, tag)
            
            vp_iu = vp_ui * (vp_i / p_u)
            vp_itu = vp_ti * vp_ui * (vp_i / (p_u * p_t))
            
            #Renormalization is necessary
            vp_iu /= vp_iu.sum()
            vp_itu /= vp_itu.sum() 
            
            tag_val = entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            return_val[tag] = tag_val
        return return_val
    
    def tag_value_gcontext(self, gamma_items = None):
        '''
        Creates a map for the value of each tag in a global context.
         
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
        vp_i = est.vect_prob_item(items)
        
        return_val = {}    
        for tag in xrange(self.est.num_tags()):
            p_t = est.prob_tag(tag)
            if p_t == 0:
                return_val[tag] = 0
                continue
            
            vp_ti = est.vect_prob_tag_given_item(items, tag)
            p_it = vp_ti * (vp_i / p_t)
            
            #Renormalization is necessary
            p_it /= p_it.sum()
            vp_i /= vp_i.sum()
            
            tag_val = entropy.kullback_leiber_divergence(p_it, vp_i)
            return_val[tag] = tag_val
        return return_val

    def prob_items_given_user(self, user, items):
        '''
        Computes the average of P(I|u)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        p_u = self.est.prob_user(user)
        vp_i = self.est.vect_prob_item(items)
        vp_ui = self.est.vect_prob_user_given_item(items, user)
        
        vp_iu = vp_ui * (vp_i / p_u)
        return vp_iu

    def prob_items_given_user_tag(self, user, tag, items):
        '''
        Computes the average of P(I|u,t)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        p_t = self.est.prob_tag(tag)
        p_u = self.est.prob_user(user)
        
        vp_i = self.est.vect_prob_item(items)
        vp_ui = self.est.vect_prob_user_given_item(items, user)
        vp_ti = self.est.vect_prob_tag_given_item(items, tag)

        vp_itu = vp_ti * vp_ui * (vp_i / (p_u * p_t))
        return vp_itu        
    
    def prob_items(self, items):
        '''
        Computes the average of P(I)
         
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        vp_i = self.est.vect_prob_item(items)
        return vp_i