#!/usr/bin/env python
# -*- encoding: utf-8
'''
Prints the tags which users posted on items for the validation and test set.
The script which creates these set does not save such files.
'''
from __future__ import division, print_function

from collections import defaultdict

from tagassess.dao.pytables.annotations import AnnotReader

import os
import plac
import sys

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
    
    val_items_fpath = os.path.join(cross_val_folder, 'user_val_items.dat')
    user_validation_items = load_dict_from_file(val_items_fpath)
    
    test_items_fpath = os.path.join(cross_val_folder, 'user_test_items.dat')
    user_test_items = load_dict_from_file(test_items_fpath)
    
    return user_items_to_filter, user_validation_items, user_test_items

def save_dict_to_file(fpath, dict_data):
    '''
    Saves a dict with sets/lists as values to a file. Each line of the file will
    be:
        key - val1, val2, val3 ...
    '''

    with open(fpath, 'w') as out_file:
        for key in dict_data:
            print(key, '-', file=out_file, end=' ')
            print(' '.join(str(x) for x in dict_data[key]), file=out_file)
     
@plac.annotations(
    db_fpath = plac.Annotation('H5 database file', type=str),
    db_name = plac.Annotation('H5 database name', type=str),
    cross_val_folder = plac.Annotation('Folder with cross validation files', 
            type=str))
def main(db_fpath, db_name, cross_val_folder):

    user_items_to_filter, user_validation_items, user_test_items = \
            load_train_test_validation(cross_val_folder)

    validation_dict = defaultdict(set)
    test_dict = defaultdict(set)
    
    with AnnotReader(db_fpath) as reader:
        reader.change_table(db_name)
        
        #Generate 50 random tags not used by any user in validation or test
        #Also creates some indexes used to define gamma items
        annotations = reader.iterate()
        for annotation in annotations:
            user = annotation['user']
            item = annotation['item']
            tag = annotation['tag']
            
            if user in user_items_to_filter \
                    and item in user_items_to_filter[user]:
                
                if item in user_validation_items[user]:
                    validation_dict[(user, item)].add(tag)
                elif item in user_test_items[user]:
                    test_dict[(user, item)].add(tag)
    
    validation_fpath = os.path.join(cross_val_folder, 
                                    'user_item_to_tags_val.dat')
    test_fpath = os.path.join(cross_val_folder, 'user_item_to_tags_test.dat')
    
    save_dict_to_file(validation_fpath, validation_dict)
    save_dict_to_file(test_fpath, test_dict)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))