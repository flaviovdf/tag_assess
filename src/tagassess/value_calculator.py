# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from itertools import ifilter
from tagassess import entropy
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.probability_estimates import SmoothEstimator
from tagassess.recommenders import ProbabilityReccomender

class ValueCalculator(object):
    '''
    Class used to compute values. 
    Contains basic value functions and filtering.
    '''
    
    def __init__(self, db_name, table, smooth_func, lambda_):
        self.db_name = db_name
        self.table = table
        self.smooth_func = smooth_func 
        self.lambda_ = lambda_
        self.est = None
        self.recc = None
        self.reader = None
        self.filter_out = None
        
    def open_reader(self, cache = True):
        '''Opens the reader, must be called before using this class'''
        self.reader = AnnotReader(self.db_name)
        self.reader.open()
        
        self.est = SmoothEstimator(self.smooth_func, self.lambda_,
                                   self._get_iterator(), cache)
        self.recc = ProbabilityReccomender(self.est)

    def close(self):
        '''Resets the value calculator'''
        if self.reader:
            self.reader.close()
            self.est = None
            self.recc = None    
            self.reader = None

    def set_filter_out(self, filter_out):
        '''Set's the row tuple pairs to disconsider'''
        self.filter_out = filter_out

    def __filt_func(self, annot):
        '''
        This method has to exist because pytables does not
        support really complex queries, like `in`.
        '''
        has_all = True
        for key in self.filter_out:
            has_all &= annot[key] in self.filter_out[key]
        
        return not has_all

    def _get_iterator(self):
        '''Get's the annotations according to `filter_out`'''
        self.reader.change_table(self.table)
        iter_table = self.reader.iterate()
        
        if self.filter_out:
            return ifilter(self.__filt_func, iter_table)
        else:
            return iter_table

    def iitem_value(self, user, items_to_compute=None):
        '''
        Creates a generator for the relevance of each item to the given user.
        This method will make use of the given `smooth_func` using the given
        `lambda_`. The generator will yield the tuple:
         (item_relevance, item).
        
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
        This method will make use of the given `smooth_func` using the given
        `lambda_`. The generator will yield the tuple:
         (tag_value, tag).
        
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
        This method will make use of the given `smooth_func` using the given
        `lambda_`. The generator will yield the tuple:
         (tag_value, tag).
         
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
    
    def get_user_tags(self, user):
        '''
        Get's the tags used by the user. These are the tags used in probability
        computations for this user
        
        Arguments
        ---------
        user: int
            The user to get tags from
        '''
        return self.est.user_tags[user]
    
    def get_tag_popularity(self, tag):
        '''
        Get's the amount of times a tag has been used in
        the collections
        
        Arguments
        ---------
        tag: int
            The tag id
        '''
        return self.est.tag_col_freq[tag]
    
    def get_tag_probability(self, tag):
        '''
        Get's the probability of tag appearing
        in the collection.
        
        Arguments
        ---------
        tag: int
            The tag id
        '''
        return self.est.prob_tag(tag)
    
    def get_item_tag_probability(self, item, tag):
        '''
        Get's the probability of tag appearing
        in the item.
        
        Arguments
        ---------
        item: int
            The item id
        tag: int
            The tag id
        '''
        return self.est.prob_tag_given_item(item, tag)
        
    def get_item_tag_popularity(self, item, tag):
        '''
        Get's the probability of tag appearing
        in the item.
        
        Arguments
        ---------
        item: int
            The item id
        tag: int
            The tag id
        '''
        if (item, tag) in self.est.item_tag_freq:
            return self.est.item_tag_freq[(item, tag)]
        else:
            return 0
    
    def get_item_probability(self, item):
        '''
        Get's the probability of item appearing
        
        Arguments
        ---------
        item: int
            The item id
        '''
        return self.est.item_col_mle[item]

    def get_item_popularity(self, item):
        '''
        Get's the popularity of an item
        
        Arguments
        ---------
        item: int
            The item id
        '''
        return round(self.est.item_col_mle[item] * self.est.n_annotations)