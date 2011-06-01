# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from itertools import ifilter
from tagassess import entropy
from tagassess.dao.annotations import AnnotReader
from tagassess.probability_estimates import SmoothedItemsUsersAsTags
from tagassess.recommenders import ProbabilityReccomender

class ValueCalculator(object):
    '''
    Class used to compute values. 
    Contains basic value functions and filtering.
    '''
    
    def __init__(self, annotation_file, table, smooth_func, lambda_):
        self.annotation_file = annotation_file
        self.table = table
        self.smooth_func = smooth_func 
        self.lambda_ = lambda_
        self.est = None
        self.recc = None
        self.reader = None
        self.filter_out = None
        
    def open_reader(self):
        '''Opens the reader, must be called before using this class'''
        self.reader = AnnotReader(self.annotation_file)
        self.reader.open_file()
        
        self.est = SmoothedItemsUsersAsTags(self.smooth_func, self.lambda_)
        self.est.open(self._get_iterator())
        self.recc = ProbabilityReccomender(self.est)

    def set_filter_out(self, filter_out):
        '''Set's the row tuple pairs to disconsider'''
        self.filter_out = filter_out

    def __filt_func(self, annot):
        '''
        This method has to exist because pytables does not
        support really complex queries, like `in`.
        '''
        methods = {'tag':lambda annot: annot.get_tag(),
                   'item':lambda annot: annot.get_item(),
                   'user':lambda annot: annot.get_user()}
        
        has_all = True
        for key in self.filter_out:
            has_all &= methods[key](annot) in self.filter_out[key]
        
        return not has_all

    def _get_iterator(self):
        '''Get's the annotations according to `filter_out`'''
        iter_table = self.reader.iterate(self.table)
        
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
            good_items = self.est.valid_items()
            items = ifilter(lambda item: item in good_items, items_to_compute)
        else:
            items = self.est.valid_items()
    
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
            items = self.est.valid_items()
                         
        est = self.est
        p_i = est.vect_prob_item(est, items)
        p_u_i = est.vect_prob_user_given_item(est, items, user)
        
        #This can be ignored, does to change rank.
        #p_u = est.prob_user(user) 
        
        if tags_to_consider:
            good_tags = self.est.valid_tags()
            tags = ifilter(lambda tag: tag in good_tags, tags_to_consider)
        else:
            tags = self.est.valid_tags()
            
        for tag in tags:
            p_t = est.prob_tag(tag)
            p_t_i = est.vect_prob_tag_given_item(est, items, tag)
            
            tag_val = entropy.kl_estimate_ucontext(p_i, p_t_i, 
                                                   p_u_i, p_t)
            
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
            items = self.est.valid_items()
                         
        est = self.est
        p_i = est.vect_prob_item(est, items)
        
        if tags_to_consider:
            good_tags = self.est.valid_tags()
            tags = ifilter(lambda tag: tag in good_tags, tags_to_consider)
        else:
            tags = self.est.valid_tags()
            
        for tag in tags:
            p_t = est.prob_tag(tag)
            p_t_i = est.vect_prob_tag_given_item(est, items, tag)
            
            tag_val = entropy.kl_estimate_gcontext(p_i, p_t_i, p_t)
            yield tag_val, tag