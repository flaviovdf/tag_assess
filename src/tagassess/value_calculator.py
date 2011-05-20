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
        self.where = None
        
    def open_reader(self):
        '''Opens the reader, must be called before using this class'''
        self.reader = AnnotReader(self.annotation_file)
        self.reader.open_file()
        
        self.est = SmoothedItemsUsersAsTags(self.annotation_file, self.table, 
                                            self.smooth_func, self.lambda_)
        self.est.open()
        self.recc = ProbabilityReccomender(self.est)

    def set_where(self, where):
        '''Set's the where query to filter the table'''
        self.where = where

    def _get_iterator(self):
        '''Get's the annotations according to `where`'''
        if self.where:
            return self.reader.iterate(self.table, self.where)
        else:
            return self.reader.iterate(self.table)

    def _items_and_user_items(self, user):
        '''
        Returns all of the items and the items of the given user
        in the annotation database.
        '''
        items = set()
        user_items = set()
        for annot in self._get_iterator():
            item = annot.get_item()
            items.add(item)
            if annot.get_user() == user:
                user_items.add(item)  
            
        return items, user_items
    
    def _tags_and_user_tags(self, user):
        '''
        Returns all of the tags and the tags of the given user
        in the annotation database.
        '''
        tags = set()
        user_tags = set()
        for annot in self._get_iterator():
            tag = annot.get_tag()
            tags.add(tag)
            if annot.get_user() == user:
                user_tags.add(tag)  
            
        return tags, user_tags
    
    def iitem_value(self, user):
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
        items, user_items = self._items_and_user_items(user)
    
        for item in items:
            relevance = self.recc.relevance(user, item)
            yield relevance, item, item in user_items
    
    def itag_value(self, user, num_to_consider=10, 
                   ignore_known_items=True):
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
        iitems = ifilter(filt, self.iitem_value(user))
        items = [(item_val[0], item_val[1]) for item_val in iitems]
                         
        if num_to_consider != -1:
            rel_items = [item[1] for item in nlargest(num_to_consider, items)]
        else:
            rel_items = [item[1] for item in items]
            
        est = self.est
        p_i = [est.prob_item(item) for item in rel_items]
        p_u_i = [est.prob_user_given_item(item, user) for item in rel_items]
        p_u = est.prob_user(user) #This can be ignored, does to change rank.
    
        tags, user_tags = self._tags_and_user_tags(user)
        for tag in tags:
            p_t = est.prob_tag(tag)
            p_t_i = [est.prob_tag_given_item(item, tag) for item in rel_items]
            
            tag_val = entropy.information_gain_estimate(p_i, p_t_i, 
                                                        p_u_i, p_t, p_u)
            
            yield (tag_val, tag, tag in user_tags)