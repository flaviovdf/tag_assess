# -*- coding: utf8
'''Module which contains functions to calculate entropy related metrics'''

from __future__ import division, print_function

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

def kl_estimate_ucontext(prob_items, prob_tag_given_items, 
                         prob_user_given_items, prob_tag, prob_user=1):
    
    '''
    Estimates how much information gain a tag adds to a user seeking relevant
    items. In formula:
    $$D_{kl}( P(\dot | u, t) || P(\dot | u))
    
    Arguments
    ---------
    prob_items: numpy array or any iterable
        The probabilities of seeing individual items P(i)

    prob_tag_given_items: numpy array or any iterable
        For each item, the probability of the tag given the item P(t|i)
    
    prob_user_given_items: numpy array or any iterable
        For each item, the probability of the user given the item P(u|i)
        
    prob_tag: float
        The probability of the individual tag under consideration
    
    prob_user: float (optional)
        The probability of the user under consideration. This parameter is
        optional since it does not affect the ranking of the estimate.
    '''
    
    np_prob_items = np.asarray(prob_items)
    np_tag_items = np.asarray(prob_tag_given_items)
    np_user_items = np.asarray(prob_user_given_items)
    
    assert len(np_prob_items) == len(np_tag_items) == len(np_user_items)
    
    np_log_diff = np.log2(np_tag_items / prob_tag)
    np_multi = np_user_items * np_tag_items * np_prob_items
    
    alpha = 1 / (prob_tag * prob_user)
    summation_part = np_multi.dot(np_log_diff)
    
    return alpha * summation_part

def kl_estimate_gcontext(prob_items, prob_tag_given_items, 
                         prob_tag):
    
    '''
    Estimates how much information gain a tag adds to a user seeking relevant
    items. In formula:
    $$D_{kl}( P(\dot | t) || P(\dot))
    
    Arguments
    ---------
    prob_items: numpy array or any iterable
        The probabilities of seeing individual items P(i)

    prob_tag_given_items: numpy array or any iterable
        For each item, the probability of the tag given the item P(t|i)
    
    prob_tag: float
        The probability of the individual tag under consideration
    '''
    
    np_prob_items = np.asarray(prob_items)
    np_tag_items = np.asarray(prob_tag_given_items)
    
    assert len(np_prob_items) == len(np_tag_items)
    
    np_log_diff = np.log2(np_tag_items / prob_tag)
    np_multi = np_tag_items * np_prob_items
    
    alpha = 1 / prob_tag
    summation_part = np_multi.dot(np_log_diff)
    
    return alpha * summation_part