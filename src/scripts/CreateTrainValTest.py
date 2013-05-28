#!/usr/bin/env python
# -*- encoding: utf-8
'''
This will separate the data in train, validation and test sets. It is expected 
that the user uses the output of this script to choose the best estimator based 
on the validation set and then perform experiments based on the test set. In
summary the script will output:

Train = The first (by date) 80% of the annotated items by each user
Validation = The following 10% of annotated items by each user
Test = The last 10% of annotated items by each user
    
Only users with more than 50 items are considered, thus at least 5 are
in the validation and test sets.
        
'''
from __future__ import division, print_function

from collections import defaultdict

from tagassess.dao.helpers import FilteredUserItemAnnotations
from tagassess.dao.pytables.annotations import AnnotReader

import os
import plac
import sys

#Percentual sizes of train, validation and test sets
PERC_ITEMS_TRAIN = .8
PERC_ITEMS_VALIDATION_TEST = .1

#At least 40 items for training
MIN_ITEMS_VAL_TRAIN = 40

#At least 10 users have to use an item or tag for it to be on the train set.
#This guarantees that items filtered for test and validation set will still
#exist on the train set. The value 10 is chosen because we want at least 5 items
#on both validation and test sets.
MIN_ITEMS_VAL_TEST = 10

def generate_indexes(reader):
    '''
    Creates some indexes to create train set and  guarantee that no item or 
    tag will be completely removed from the train set.
    '''
    user_to_items = defaultdict(list) #a list is needed to maintain date order
    user_to_tags = defaultdict(set) 
    item_to_tags = defaultdict(set)
    for annotation in reader.iterate():
        user = annotation['user']
        item = annotation['item']
        tag = annotation['tag']
        
        if item not in user_to_items[user]:
            user_to_items[user].append(item)
            
        user_to_tags[user].add(tag)
        item_to_tags[item].add(tag)

    return user_to_items, user_to_tags, item_to_tags

def create_train_test_validation(reader):
    '''
    Separates the annotation in a train, test and validation sets. The code
    will work on users with at lest 50 items. For each of these users the code 
    considers the first 80% (40 or more) by date for train, the following 
    10% for validation and for last 10% for test. The catch is that when 
    filtering items from users for the validation and test sets, some items may
    be removed from the trace completely. Thus, we only consider users which we
    can filter items which will not be removed completely from the trace, that 
    is, users which we can guarantee that their removed items exist in the 
    train set.
    
    This code considers that annotations are already sorted by date for each
    user. This sort is from old to new (ascending date order). 
    '''
    
    user_to_items, user_to_tags, item_to_tags = \
            generate_indexes(reader)
    
    #the next code will generated candidate pairs to be left out of train
    user_items_to_filter = defaultdict(set)
    user_validation_items = defaultdict(set)
    user_validation_tags = defaultdict(set)
    user_test_items = defaultdict(set)
    user_test_tags = defaultdict(set)
    
    #Generate users which belong to training set and users which can be 
    #considered for testing. The testing users are those with at least
    #10 items (minimum of 5 for validation set and 5 for train set).
    candidate_users = set()
    train_users_items = set()
    train_users_tags = set()
    for user in user_to_items:
        if len(user_to_items[user]) < MIN_ITEMS_VAL_TRAIN + MIN_ITEMS_VAL_TEST:
            train_users_items.update(user_to_items[user])
            train_users_tags.update(user_to_tags[user])
        else:
            candidate_users.add(user)

    for user in candidate_users:

        #num items to remove for this user
        num_item_val_test = 2 * \
                int(PERC_ITEMS_VALIDATION_TEST * len(user_to_items[user]))
        
        #Negative indexing begins backwards, thus we get the last items
        val_test_items = user_to_items[user][-num_item_val_test:]
        assert len(val_test_items) >= MIN_ITEMS_VAL_TEST
        
        #For each validation and test item and also for each tag on those
        #items, we have to guarantee that such item and tag exists in the
        #training set
        to_remove = []
        for item in val_test_items:
            #not a good candidate item
            if item not in train_users_items:
                to_remove.append(item)
                continue
            
            for tag in item_to_tags[item]:
                #also not good, a tag may be removed from the trace
                if tag not in train_users_tags:
                    to_remove.append(item)
                    break

        #Remove non good items from candidates
        for item in to_remove:
            val_test_items.remove(item)
                
        #Do we have at least 10 items? 5 for validation and for test? If not,
        #this user will not be in our evaluation
        num_val_test = len(val_test_items)
        if num_val_test >= MIN_ITEMS_VAL_TEST:
            half = num_val_test // 2
            
            #Now we can populate the dictionaries with the test and validation
            #items and tags
            for i in xrange(num_val_test):
                item = val_test_items[i]
                user_items_to_filter[user].add(item)
                
                if i < half:
                    item_dict = user_validation_items
                    tags_dict = user_validation_tags
                else:
                    item_dict = user_test_items
                    tags_dict = user_test_tags
                
                item_dict[user].add(item)
                for tag in item_to_tags[item]:
                    tags_dict[user].add(tag)
        
    return user_items_to_filter, user_validation_items, user_validation_tags, \
             user_test_items, user_test_tags

def sanity_check(reader, user_items_to_filter):
    '''
    A simple sanity check to verify that we did not delete any
    user, item or tag from the trace.
    '''
    
    users = set()
    items = set()
    tags = set()
    for annotation in reader.iterate():
        user = annotation['user']
        item = annotation['item']
        tag = annotation['tag']
        users.add(user)
        items.add(item)
        tags.add(tag)
    
    filtered = FilteredUserItemAnnotations(user_items_to_filter)
    filtered_users = set()
    filtered_items = set()
    filtered_tags = set()
    for annotation in filtered.annotations(reader.iterate()):
        user = annotation['user']
        item = annotation['item']
        tag = annotation['tag']
        
        assert user in users
        assert item in items
        assert tag in tags
        
        if user in user_items_to_filter:
            assert item not in user_items_to_filter[user]
        
        filtered_users.add(user)
        filtered_items.add(item)
        filtered_tags.add(tag)
    
    assert len(filtered_users) == len(users)
    assert len(filtered_items) == len(items)
    assert len(filtered_tags) == len(tags)
    
    assert len(filtered_users.symmetric_difference(users)) == 0
    assert len(filtered_items.symmetric_difference(items)) == 0
    assert len(filtered_tags.symmetric_difference(tags)) == 0
    
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
    output_folder = plac.Annotation('Output folder path', type=str))
def main(db_fpath, db_name, output_folder):
    
    assert os.path.isdir(output_folder)
    assert len(os.listdir(output_folder)) == 0
    
    with AnnotReader(db_fpath) as reader:
        reader.change_table(db_name)
        
        user_items_to_filter, user_validation_items, user_validation_tags, \
             user_test_items, user_test_tags = \
                    create_train_test_validation(reader)
        
        #Sanity check
        sanity_check(reader, user_items_to_filter)
        
        filter_fpath = os.path.join(output_folder, 'user_item_filter.dat')
        save_dict_to_file(filter_fpath, user_items_to_filter)
        
        val_items_fpath = os.path.join(output_folder, 'user_val_items.dat')
        save_dict_to_file(val_items_fpath, user_validation_items)
        
        val_tags_fpath = os.path.join(output_folder, 'user_val_tags.dat')
        save_dict_to_file(val_tags_fpath, user_validation_tags)
        
        test_items_fpath = os.path.join(output_folder, 'user_test_items.dat')
        save_dict_to_file(test_items_fpath, user_test_items)
        
        test_tags_fpath = os.path.join(output_folder, 'user_test_tags.dat')
        save_dict_to_file(test_tags_fpath, user_test_tags)
        
if __name__ == '__main__':
    sys.exit(plac.call(main))
