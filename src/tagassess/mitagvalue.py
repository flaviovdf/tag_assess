# -*- coding: utf8
'''
MUTUAL INFORMATION TAG VALUE
============================

Implementation of the tag assessment method described on the paper:
http://dx.doi.org/10.1109/SocialCom.2010.69
'''
from __future__ import print_function, division

import numpy as np

def __assert_good_probs(prob_array):
    '''
    Utility function to guarantee that arrays: sum to 1 and have all
    values x in prob_array : 0 <= x <= 1

    Arguments
    ---------
    prob_array: a numpy one dimensional array
    '''
    #Filter elements which are 0 <= x <= 1
    good_elements = prob_array[(prob_array >= 0) & (prob_array <= 1)]
    psum = np.sum(prob_array)
    
    eq_length = len(good_elements) == len(prob_array)
    return psum == 1.0 and eq_length

def estimate_prob_item_for_user_tag(item, tag, user, annotation_it):
    '''
    Given an item a user and a tag, this method computes the probability
    of the item being relevant given the user and tag : p(i|t,u). The 
    `annotation_it` is a iterator which should yield `Annotation` objects.
    The `item` parameter is an item id which should exist in the iterator, 
    the same holds for the `tag` id. Although we call the argument `user`
    this can be any set of tag ids.
    
    Arguments
    ---------
    item: int
        A item id
    tag: int
        A user id
    user: set
        A set of tag ids which represents the user past tags
    annotation_it:
        The iterator for annotations to consider in the calculation
    
    See also
    --------
    tagassess.dao.AnnotReader, tagassess.dao.Annotation
    '''
    pass

def entropy(x_probabilities):
    '''
    Calculates the entropy (H) of the input vector which
    represents some random variable X.

    Arguments
    ---------
    x_probabilities: numpy array or any iterable
        Array with the individual x_probabilities. Values must be 0 <= x <=1
    '''

    probs = np.asarray(x_probabilities)
    assert __assert_good_probs(probs)

    return -1 * np.add.reduce(x_probabilities * np.log2(probs))

def norm_mutual_information(probabilities_x, probabilities_xy):
    '''
    Calculates the normalized mutual information between the
    random variables (X and X|Y):

    Arguments
    ---------
    probabilities_x: numpy array or any iterable
        Array with the individual probabilities X. Values must be 0 <= x <= 1

    probabilities_xy: numpy array or any iterable
        Array with the individual probabilities for X|Y. Values must be 0 <= x <= 1
    '''

    h_x = entropy(probabilities_x)
    h_xy = entropy(probabilities_xy)

    normalized_mi = 0
    if h_x > 0 and h_xy > 0:
        normalized_mi = 1 - (h_x - h_xy) / h_x
        
    return normalized_mi