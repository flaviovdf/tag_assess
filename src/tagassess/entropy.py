# -*- coding: utf8
'''Module which contains functions to calculate entropy related metrics'''

from __future__ import division, print_function

import numpy as np
import numpy.ma as ma

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
    psum = prob_array.sum()
    
    eq_length = len(good_elements) == len(prob_array)
    return psum >= 0.9999999999 and psum <= 1.0000000001 and eq_length

def mask_zeros(func):
    '''
    Defines a decorator which will:
        1. Convert arguments to numpy arrays
        2. Mask zeros (zero prob is ignored in entropy calculations)
        3. Check if probabilities are valid: sum == 1 and values in [0,1]
    '''
    
    def decorator(*args):
        '''The decorator function'''
        array_args = []
        for arg in args:
            array = np.asarray(arg)
            masked_array = ma.masked_array(array, array == 0) #Mask zeros.
            array_args.append(masked_array)
            __assert_good_probs(masked_array)
        return func(*array_args)
    return decorator

@mask_zeros
def entropy(probabilities_x):
    '''
    Calculates the entropy (H) of the input vector which
    represents some random variable X.

    Arguments
    ---------
    probabilities_x: numpy array or any iterable
        Array with the individual probabilities_x. Values must be 0 <= x <=1
    '''
    return -1 * np.add.reduce(probabilities_x * np.log2(probabilities_x))

@mask_zeros
def mutual_information(probabilities_x, probabilities_xy):
    '''
    Calculates the mutual information between the
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
    return h_x - h_xy

@mask_zeros
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

@mask_zeros
def kullback_leiber_divergence(probabilities_p, probabilities_q):
    '''
    Calculates the Kullback-Leiber divergence between the distributions
    of two random variables.

    $$ D_{kl}(P(X) || Q(X)) = \sum_{x \in X) p(x) * log(\frac{p(x)}{q(x)}) $$

    Arguments
    ---------
    probabilities_p: numpy array or any iterable
        Array with the individual probabilities P. Values must be 0 <= x <= 1

    probabilities_q: numpy array or any iterable
        Array with the individual probabilities for Q. Values must be 0 <= x <= 1
    '''
    #n * log(n / 0) = inf (definition of kullback leiber)
    #Since we have arrays which mask zeros. We get the 
    #elements in Q where P is not masked. If any of these are
    #also masked, we have: n * log(n / 0)
    is_val_pos = ~probabilities_q.mask[~probabilities_p.mask]
    if not is_val_pos.all():
        return np.float('inf')
    
    log_part = np.log2(probabilities_p) - np.log2(probabilities_q)
    return np.add.reduce(probabilities_p * log_part)