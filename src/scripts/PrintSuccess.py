#!/usr/bin/env python
# -*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict
from scipy.stats import norm
from scipy.stats import t

import glob
import numpy as np
import os
import plac
import sys
import tables

def confidence_interval(data, confidence, axis=None):
    """
    Determines the half of the confidence interval size
    for some data. The confidence interval is mean +- return values.
    
    >>> data = [8.0, 7.0, 5.0, 9.0, 9.5, 11.3, 5.2, 8.5]
    >>> confidence_interval(data, 0.95)
    1.6772263663789651
    """
    
    a = np.asanyarray(data)
    if axis is None:
        n = a.size
    else:
        n = a.shape[axis]
        
    std = np.std(a, axis=axis)
    
    # calls the inverse CDF of the Student's t
    # distribution
    if n <= 30:
        h = std * t_table(n-1, confidence) / np.sqrt(n)
    else:
        h = std * z_table(confidence) / np.sqrt(n)
    return h

def t_table(freedom, confidence):
    """
    Looks up the t_table (i.e. calls the inverse CDF of the 
    t-student distribution).
    
    >>> t_table(1, 0.95)
    12.706204736432099
    >>> t_table(9, 0.90)
    1.8331129326536333
    >>> t_table(8, 0.99)
    3.3553873313333957
    >>> t_table(10, 0.85)
    1.5592359332425447
    """
    return t.ppf((1+confidence)/2.,freedom)

def z_table(confidence):
    """
    Looks up the z_table (i.e. calls the inverse CDF of the 
    normal distribution).
    
    >>> z_table(0.95)
    1.959963984540054
    >>> z_table(0.90)
    1.6448536269514722
    >>> z_table(0.99)
    2.5758293035489004
    >>> z_table(0.85)
    1.4395314709384561
    """
    return norm.ppf((1+confidence)/2.)

def load_dict_from_file(fpath):
    '''Loads dictionary from file'''

    return_val = {}
    with open(fpath) as in_file:
        for line in in_file:
            spl = line.split('-')
            
            key_split = spl[0].strip().split(',')
            if len(key_split) == 1:
                key = int(key_split[0].strip())
            else:
                key_0 = int(key_split[0][1:].strip())
                key_1 = int(key_split[1][:-1].strip())
                key = (key_0, key_1)
            
            value = set(int(x.strip()) for x in spl[1].split())
            return_val[key] = value

    return return_val

def load_train_test_validation(cross_val_folder):
    '''Loads cross validation dictionaries used for the experiment'''

    filter_fpath = os.path.join(cross_val_folder, 'user_item_filter.dat')
    user_items_to_filter = load_dict_from_file(filter_fpath)

    val_items_fpath = os.path.join(cross_val_folder, 'user_val_items.dat')
    user_validation_items = load_dict_from_file(val_items_fpath)

    val_tags_fpath = os.path.join(cross_val_folder, 'user_val_tags.dat')
    user_validation_tags = load_dict_from_file(val_tags_fpath)

    user_item_to_tag_val_fpath = os.path.join(cross_val_folder, 
            'user_item_to_tags_val.dat')
    user_item_to_tags_val = load_dict_from_file(user_item_to_tag_val_fpath)

    return user_items_to_filter, user_validation_items, user_validation_tags, \
            user_item_to_tags_val

def succ_at_k(gamma, probs, relevant, k=10):
    '''
    Success @ k is the number of items at least ONE item appeared in the 
    topk list
    '''
    relevant = set(relevant)
    revese_sorted_idx = probs.argsort()[::-1]
    top_k = set(gamma[revese_sorted_idx][:k])

    if len(relevant.intersection(top_k)) > 0:
        return 1
    else:
        return 0

def main(cv_folder, param_folder, estimator):

    np.seterr(all='raise') #raise exception for any warning

    assert estimator in ['lda', 'smooth']

    user_items_to_filter, user_validation_items, user_validation_tags, \
            user_item_to_tag_val = load_train_test_validation(cv_folder)

    users_fpaths = glob.glob(os.path.join(param_folder, 'user-*.h5'))
    
    success_piu = []
    success_pitu = []
    success_piqu = []

    for user_fpath in users_fpaths:
        user_id = int(user_fpath.split('-')[-1].split('.')[0])
        
        h5file = tables.openFile(user_fpath, mode='r')
        child_nodes = h5file.iterNodes(h5file.root)
        
        gamma = h5file.getNode(h5file.root, 'gamma').read()
        piu = h5file.getNode(h5file.root, 'piu').read()
            
        pitus = {}
        for child_node in child_nodes:
            if 'pitu' in child_node.name:
                tag_id = int(child_node.name.split('_')[-1])
                if tag_id in user_validation_tags[user_id]:
                    pitus[tag_id] = child_node.read()
        
        #Estimate query value (we use log(prb + 1) to avoid underflows). Not a problem
        #since this is just for ranking
        piqus = []
        if estimator == 'smooth': #For smooth it is just the product
            for item_id in user_validation_items[user_id]:
                piqu = 0
                for tag_id in user_item_to_tag_val[(user_id, item_id)]:
                    piqu += np.log(pitus[tag_id] + 1)
                
                piqus.append(piqu)
        else: #For lda we have to consider the model
            for item_id in user_validation_items[user_id]:
                piqu = 0
                for tag_id in user_item_to_tag_val[(user_id, item_id)]:
                    piqu += np.log(pitus[tag_id] + 1) - np.log(piu + 1)
            
                piqu += np.log(piu + 1)
                piqus.append(piqu)

        relevant = user_validation_items[user_id]
        success_piu.append(succ_at_k(gamma, piu, relevant))
        
        sum_success_tags = 0
        for tag_id in pitus:
            sum_success_tags += succ_at_k(gamma, pitus[tag_id], relevant)
        success_pitu.append(sum_success_tags / len(pitus))
        
        sum_success_queries = 0
        for piqu in piqus:
            sum_success_queries += succ_at_k(gamma, piqu, relevant)
        success_piqu.append(sum_success_queries / len(piqus))

        h5file.close()
    
    mean_piu = np.mean(success_piu)
    ci_piu = confidence_interval(success_piu, .95)
    print('S@10 sorted P(I|u) vs relevant = ', mean_piu, '+-', ci_piu)


    mean_pitu = np.mean(success_pitu)
    ci_pitu = confidence_interval(success_pitu, .95)
    print('S@10 sorted avg P(I|t,u) vs relevant = ', mean_pitu, '+-', ci_pitu)

    mean_piqu = np.mean(success_piqu)
    ci_piqu = confidence_interval(success_piqu, .95)
    print('S@10 sorted avg P(I|q,u) vs relevant = ', mean_piqu, '+-', ci_piqu)

if __name__ == '__main__':
    plac.call(main)
