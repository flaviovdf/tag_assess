#!/usr/bin/env python
# -*- encoding: utf-8
from __future__ import division, print_function

from tagassess.dao.helpers import FilteredUserItemAnnotations
from tagassess.dao.pytables.annotations import AnnotReader
from tagassess.probability_estimates.precomputed import PrecomputedEstimator
from tagassess.value_calculator import ValueCalculator

import numpy as np
import os
import plac
import sys

def run_exp(user_validation_tags, user_test_tags, est, value_calc):
    
    print('#user', 'tag', 'rho', 'dkl', 'value', 'hidden_tag')
    for user in est.user_to_tags.keys():
        
        tags = est.tags_for_user(user)
        gamma = est.gamma_for_user(user)
        
        #Remove validation tags. The script focuses on test tags
        tags_to_compute = []
        for tag in tags:
            if tag not in user_validation_tags[user]:
                tags_to_compute.append(tag)
                
        tags_to_compute = np.asanyarray(tags_to_compute)
        values = value_calc.tag_value_personalized(user, gamma, tags_to_compute, 
                        True)
        
        for tag_idx, tag in enumerate(tags_to_compute):
            hidden = tag in user_test_tags[user]
            print(user, tag, values[tag_idx, 0], values[tag_idx, 1], 
                  values[tag_idx, 2], hidden)

def load_dict_from_file(fpath):
    '''Loads dictionary from file'''
    
    return_val = {}
    with open(fpath) as in_file:
        for line in in_file:
            spl = line.split('-')
            key = int(spl[0].strip())
            value = set(int(x.strip()) for x in spl[1].split())
            
            return_val[key] = value
            
    return return_val

def load_train_test_validation(cross_val_folder):
    '''Loads cross validation dictionaries used for the experiment'''
    
    filter_fpath = os.path.join(cross_val_folder, 'user_item_filter.dat')
    user_items_to_filter = load_dict_from_file(filter_fpath)
    
    val_tags_fpath = os.path.join(cross_val_folder, 'user_val_tags.dat')
    user_validation_tags = load_dict_from_file(val_tags_fpath)
    
    test_tags_fpath = os.path.join(cross_val_folder, 'user_test_tags.dat')
    user_test_tags = load_dict_from_file(test_tags_fpath)
    
    return user_items_to_filter, user_validation_tags, user_test_tags
                
@plac.annotations(
    db_fpath = plac.Annotation('H5 database file', type=str),
    db_name = plac.Annotation('H5 database name', type=str),
    cross_val_folder = plac.Annotation('Folder with cross validation files', 
            type=str),
    probs_folder = plac.Annotation('Probabilities Folder', type=str))
def main(db_fpath, db_name, cross_val_folder, probs_folder):
    
    #get cross validation dicts
    user_items_to_filter, user_validation_tags, user_test_tags = \
            load_train_test_validation(cross_val_folder)

    with AnnotReader(db_fpath) as reader:
        reader.change_table(db_name)
        
        annot_filter = FilteredUserItemAnnotations(user_items_to_filter)
        annotations = annot_filter.annotations(reader.iterate())
        
        est = PrecomputedEstimator(probs_folder)
        value_calc = ValueCalculator(est, annotations)
        
        run_exp(user_validation_tags, user_test_tags, est, value_calc)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))