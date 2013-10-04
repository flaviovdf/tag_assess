#!/usr/bin/env python
# -*- encoding: utf-8
from __future__ import division, print_function

from tagassess.dao.helpers import FilteredUserItemAnnotations
from tagassess.dao.pytables.annotations import AnnotReader
from tagassess.index_creator import create_occurrence_index
from tagassess.probability_estimates.precomputed import PrecomputedEstimator

import os
import plac
import sys

def get_baselines(annot_filter, reader, user_to_tags):
    
    annotations = annot_filter.annotations(reader.iterate())
    user_to_item = create_occurrence_index(annotations, 'user', 'item')
    
    annotations = annot_filter.annotations(reader.iterate())
    item_to_tags = create_occurrence_index(annotations, 'item', 'tag')
    
    overlap = {}
    for user in user_to_tags:
        for item in user_to_item:
            for tag in item_to_tags[item]:
                if (user, tag) not in overlap:
                    overlap[user, tag] = 0
                    
                if tag not in user_to_tags[user]:
                    overlap[user, tag] += 1
    
    idf = {}
    annotations = annot_filter.annotations(reader.iterate())
    for annot in annotations:
        tag = annot['tag']
        if tag not in idf:
            idf[tag] = 0
            
        idf[tag] += 1
    
    for tag in idf.keys():
        idf[tag] = 1.0 / idf[tag] 
    
    return idf, overlap

def run_exp(user_validation_tags, user_test_tags, est, annot_filter, reader):
    
    user_to_tags = {}
    for user in est.get_valid_users():
        #Remove validation tags. The script focuses on test tags
        tags_to_compute = []
        tags = est.tags_for_user(user)
        for tag in tags:
            if tag not in user_validation_tags[user]:
                tags_to_compute.append(tag)
                
        user_to_tags[user] = tags_to_compute
    
    idf, overlap = get_baselines(annot_filter, reader, user_to_tags)

    print('#user', 'tag', 'pop(1/idf)', 'idf', 'overlap', 
          'overlap*idf', 'hidden_tag')
    for user in est.get_valid_users():
        tags = user_to_tags[user]
        
        for tag in tags:
            hidden = tag in user_test_tags[user]
            print(user, tag, 1.0 / idf[tag], idf[tag], overlap[user, tag],
                  overlap[user, tag] * idf[tag], hidden)

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
        est = PrecomputedEstimator(probs_folder)
        run_exp(user_validation_tags, user_test_tags, est, annot_filter, reader)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))