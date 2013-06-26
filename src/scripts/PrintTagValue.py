#!/usr/bin/env python
# -*- encoding: utf-8
'''
This script is used to will print a table with tag values for each user in
the TESTING set. Parameters must have been learned in the GridSearch script
using the training set.

The output is printed in stdout.
'''
from __future__ import division, print_function

from collections import defaultdict

from random import seed
from random import shuffle

from tagassess.dao.helpers import FilteredUserItemAnnotations
from tagassess.dao.pytables.annotations import AnnotReader
from tagassess.probability_estimates.helpers import create_bayes_estimator
from tagassess.probability_estimates.helpers import create_lda_estimator
from tagassess.value_calculator import ValueCalculator

import numpy as np
import os
import plac
import sys

NUM_RANDOM_TAGS = 50

def run_exp(user_items_to_filter, user_test_tags, user_to_item, num_items, 
            random_tags, value_calc):
    
    print('#user', 'tag', 'rho', 'dkl', 'value', 'hidden_tag')
    for user in user_items_to_filter:
        gamma_items = [item for item in xrange(num_items) 
                            if item not in user_to_item[item]]
        gamma_items = np.asarray(gamma_items)
        
        tags_for_user = set()
        for tag in random_tags:
            tags_for_user.add(tag)
        
        for tag in user_test_tags[user]:
            tags_for_user.add(tag)
        
        tags = np.asarray([tag for tag in tags_for_user])
        values = value_calc.tag_value_personalized(user, gamma_items, tags, 
                True)
        
        for tag_idx in range(tags.shape[0]):
            tag = tags[tag_idx]
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
    param_value = plac.Annotation('Parameter Value', type=float),
    est_name = plac.Annotation('Estimator to perform grid search', type=str, 
            choices=['lda', 'smooth'], kind='option'),
    rand_seed = plac.Annotation('Random seed to use (None = default seed)',
            type=int, kind='option'),
    num_cores = plac.Annotation('Number of cores to use', type=int))
def main(db_fpath, db_name, cross_val_folder, param_value, est_name, 
         rand_seed=None, num_cores=-1):
    '''Dispatches jobs in multiple cores'''
    
    seed(rand_seed)
    
    #get cross validation dicts
    user_items_to_filter, user_validation_tags, user_test_tags = \
            load_train_test_validation(cross_val_folder)

    #all tags used by all users. Used o create a random set of tags excluding 
    #these ones
    used_tags = set()
    for user in user_items_to_filter:
        used_tags.update(user_validation_tags[user])
        used_tags.update(user_test_tags[user])
    
    with AnnotReader(db_fpath) as reader:
        reader.change_table(db_name)
        
        annot_filter = FilteredUserItemAnnotations(user_items_to_filter)
        
        #Generate 50 random tags not used by any user the test set
        #Also creates some indexes used to define gamma items
        annotations = annot_filter.annotations(reader.iterate())
        user_to_item = defaultdict(set)
        items = set()
        tags = set()
        random_tags = []
        for annotation in annotations:
            user = annotation['user']
            item = annotation['item']
            tag = annotation['tag']
            
            user_to_item[user].add(item)
            items.add(item)
            tags.add(tag)
            
            if tag not in used_tags and tag not in random_tags:
                random_tags.append(tag)
        
        shuffle(random_tags)
        random_tags = random_tags[:NUM_RANDOM_TAGS]    
        
        #Gets number of tags and items
        num_items = len(items)
        num_tags = len(tags)
        
        #Create estimator
        annotations = annot_filter.annotations(reader.iterate())
        if est_name == 'lda':
            est = create_lda_estimator(annotations, param_value, 
                num_items, num_tags)
        else:
            est = create_bayes_estimator(annotations, param_value)

        annotations = annot_filter.annotations(reader.iterate())
        value_calc = ValueCalculator(est, annotations)
        
        run_exp(user_items_to_filter, user_test_tags, user_to_item, num_items, 
                random_tags, value_calc)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))