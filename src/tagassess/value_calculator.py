# -*- coding: utf8
'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from heapq import nlargest
from tagassess import entropy
from tagassess.dao.annotations import AnnotReader
from tagassess.probability_estimates import SmoothedItemsUsersAsTags
from tagassess.recommenders import ProbabilityReccomender

def _items_and_user_items(annotation_file, table, user):
    '''
    Returns all of the items and the items of the given user
    in the annotation database.
    '''
    with AnnotReader(annotation_file) as reader:
        items = set(a.get_item() for a in reader.iterate(table))
        user_sel = 'USER == ' + str(user)
        user_items = set(a.get_item() for a in reader.iterate(table, user_sel))
    
    return items, user_items

def _tags_and_user_tags(annotation_file, table, user):
    '''
    Returns all of the tags and the tags of the given user
    in the annotation database.
    '''
    with AnnotReader(annotation_file) as reader:
        tags = set(a.get_tag() for a in reader.iterate(table))
        user_sel = 'USER == ' + str(user)
        user_tags = set(a.get_tag() for a in reader.iterate(table, user_sel))

    return tags, user_tags

def iitem_value(annotation_file, table, user, smooth_func, lambda_, 
                ignore_known_items=True):
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
    #TODO: Not the most efficient implementation. Change if needed.
    #It all be done in one loop, we organize with 3 for testing.
    items, user_items = _items_and_user_items(annotation_file, table, user)
    
    estimator = SmoothedItemsUsersAsTags(annotation_file, table, 
                                         smooth_func, lambda_)
    estimator.open()
    reccomender = ProbabilityReccomender(estimator)

    #Can also be optimized    
    to_iter = items if not ignore_known_items else items.difference(user_items)
    for item in to_iter:
        relevance = reccomender.relevance(user, item)
        yield (relevance, item, item in user_items)

def itag_value(annotation_file, table, user, smooth_func, lambda_,
               num_to_consider=10, ignore_known_items=True, items=None):
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
    if not items:
        iitems = iitem_value(annotation_file, table, user, smooth_func, 
                             lambda_, ignore_known_items)
        items = [(tup[0], tup[1]) for tup in iitems]
                     
    if num_to_consider != -1:
        rel_items = [item[1] for item in nlargest(num_to_consider, items)]
    else:
        rel_items = [item[1] for item in items]
        
    est = SmoothedItemsUsersAsTags(annotation_file, table, 
                                   smooth_func, lambda_)
    est.open()
    
    p_items = [est.prob_item(item) for item in rel_items]
    p_user_item = [est.prob_user_given_item(item, user) for item in rel_items]
    p_user = est.prob_user(user) #This can be ignored, does to change rank.

    tags, user_tags = _tags_and_user_tags(annotation_file, table, user)    
    for tag in tags:
        p_tag = est.prob_tag(tag)
        p_tag_item = [est.prob_tag_given_item(item, tag) for item in rel_items]
        
        tag_val = entropy.information_gain_estimate(p_items, p_tag_item, 
                                                    p_user_item, p_tag, p_user)
        
        yield (tag_val, tag, tag in user_tags)