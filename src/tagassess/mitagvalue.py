# -*- coding: utf8
'''
MUTUAL INFORMATION TAG VALUE
============================

Implementation of the tag assessment method described on:
http://dx.doi.org/10.1109/SocialCom.2010.69
'''
from __future__ import print_function, division

import numpy as np

def __assert_good_probs(prob_array):
    '''
    Utility function to guarantee that arrays: sum to 1 and have all
    values x : 0 <= x <= 1

    Arguments
    ---------
    prob_array: a numpy one dimensional array
    '''
    geq = prob_array[prob_array >= 0] #Filter elements which are >= 0
    leq = prob_array[prob_array <= 1] #Filter <= 1
    psum = np.sum(prob_array)
    return len(geq) == len(leq) == len(prob_array) and psum == 1.0

def entropy(probabilities):
    '''
    Calculates the entropy (H) of the input vector.

    Arguments
    ---------
    probabilities: numpy array or any iterable
        Array with the individual probabilities. Values must be 0 <= x <=1
    '''

    probs = np.asarray(probabilities)
    assert __assert_good_probs(probs)

    return -np.add.reduce(probabilities * np.log2(probs))

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

    if h_x > 0:
        return 1 - (h_x - h_xy) / h_x
    else:
        return 0
