# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from heapq import nlargest
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
         (item_relevance, item, True if user has tagged item and False otherwise).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        if items_to_compute:
            items = items_to_compute
        else:
            items = self.est.item_col_freq.keys()
    
        for item in items:
            relevance = self.recc.relevance(user, item)
            yield relevance, item
    
    def itag_value(self, user, num_to_consider=10, 
                   ignore_known_items=True, items_to_compute=None,
                   tags_to_consider=None):
        '''
        Creates a generator for the value of each tag to the given user.
        This method will make use of the given `smooth_func` using the given
        `lambda_`. The generator will yield the tuple:
         (tag_value, tag, True if user has used the tag and False otherwise).
        
        See also
        --------
        tagassess.smooth
        tagassess.probability_estimates
        '''
        filt = lambda item_val: not (ignore_known_items and item_val[2])
        iitems = ifilter(filt, self.iitem_value(user, items_to_compute))
        items = [(item_val[0], item_val[1]) for item_val in iitems]
                         
        if num_to_consider != -1:
            rel_items = [item[1] for item in nlargest(num_to_consider, items)]
        else:
            rel_items = [item[1] for item in items]
            
        est = self.est
        p_i = [est.prob_item(item) for item in rel_items]
        p_u_i = [est.prob_user_given_item(item, user) for item in rel_items]
#        p_u = est.prob_user(user) #This can be ignored, does to change rank.
        
        if tags_to_consider:
            tags_with_nonz = self.est.tag_col_freq
            tags = ifilter(lambda tag: tag in tags_with_nonz, tags_to_consider)
        else:
            tags = self.est.tag_col_freq.keys()
            
        for tag in tags:
            p_t = est.prob_tag(tag)
            p_t_i = [est.prob_tag_given_item(item, tag) for item in rel_items]
            
            tag_val = entropy.information_gain_estimate(p_i, p_t_i, 
                                                        p_u_i, p_t)
            
            yield tag_val, tag