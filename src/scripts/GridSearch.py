#!/usr/bin/env python
# -*- encoding: utf-8
'''
This script is used to search best parameters for the probability estimators
which we consider in our experiments. It will use the data separated in train, 
validation and test sets. Probabilities p(i|u) and p(i|t,u) will be computed
for the validation and test set. It is expected that the user uses the output
of this script to choose the best estimator based on the validation set and then
perform experiments based on the test set.
'''
from __future__ import division, print_function

from collections import defaultdict

from itertools import combinations

from random import seed
from random import shuffle

from tagassess.dao.helpers import FilteredUserItemAnnotations
from tagassess.dao.pytables.annotations import AnnotReader
from tagassess.probability_estimates.helpers import create_bayes_estimator
from tagassess.probability_estimates.helpers import create_lda_estimator

import numpy as np
import multiprocessing
import os
import plac
import sys
import tables

#Parameter values considered
#For the Bayes smooth, the paper plots -log10([2 .. 6]), thus values are
#the x = -log10([2 .. 6]), which is 10 ** [2 .. 6].
#We also vary the number of tags in the user profile
SMOOTH_PARAMS = {'p1_lambda':10 ** (-np.arange(2, 6.01, .5)),
                 'p2_fract_tags':[.25, .50, .75, 1]}

#For LDA estimator only gamma varies, other parameters are based on dataset.
LDA_GAMMA_PARAMS = {'p1_gamma':[0.25, 0.5, 0.001, 0.01, 0.1, 25, 50],
                    'p2_num_topics':[150, 250, 300]}

NUM_RANDOM_TAGS = 50

def run_exp(user_items_to_filter, user_validation_tags, user_test_tags, 
        user_to_item, num_items, random_tags, est, output_folder, save_lhood):
    '''Computes probabilities for one user and saves results to files'''
    
    #Save train data if necessary
    if save_lhood:
        chain = est.chain_likelihood()
        log_likelihood = est.log_likelihood()
        
        train_fpath = os.path.join(output_folder, 'train-data.h5')
        train_h5file = tables.openFile(train_fpath, mode='w')
        
        train_h5file.createArray(train_h5file.root, 'chain', chain)
        train_h5file.createArray(train_h5file.root, 'likelihood',
                np.array([log_likelihood]))

        train_h5file.close()

    #Run experiment
    for user in user_items_to_filter:
        
        user_fpath = os.path.join(output_folder, 'user-%d.h5' % user)
        user_h5file = tables.openFile(user_fpath, mode='w')
        
        gamma_items = [item for item in xrange(num_items) 
                            if item not in user_to_item[item]]
        
        gamma_items = np.asarray(gamma_items)
        user_h5file.createArray(user_h5file.root, 'gamma', gamma_items)
        
        probs_i_given_u = est.prob_items_given_user(user, gamma_items)
        user_h5file.createArray(user_h5file.root, 'piu', probs_i_given_u)

        tags_for_user = set()
        for tag in random_tags:
            tags_for_user.add(tag)
        
        for tag in user_validation_tags[user]:
            tags_for_user.add(tag)
        
        for tag in user_test_tags[user]:
            tags_for_user.add(tag)
        
        for tag in tags_for_user:
            probs_i_given_u_t = est.prob_items_given_user_tag(user, tag, 
                gamma_items)
            user_h5file.createArray(user_h5file.root, 'pitu_tag_%d' % tag, 
                    probs_i_given_u_t)
            
        user_h5file.close()

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

def run_one(args):
    '''
    This method will be run by parallel processes. Basically, it is the
    main method for each possible parameter being tested. It will work as
    follows:
    
    1. Loads train, validation and test separation from files
    
    2. Values of p(i|u) are computed for the gamma items set for each user
       based on the train set. Gamma items is just every item excluding the
       user items.
       
    3. Computes p(i|t,u) for a set of tags gamma items for each user. The set
       of tags is composed of the previous user tags (those on the test set), 
       the tags which were used on the validation set, the tags used on the
       train set and 50 random tags not previously used by the user.
    
    4. Saves p(i|u) and p(i|t,u) for items and tags considered above on the
       output folder. This provides sufficient information for choosing the best
       estimator (on the validation set) and performing further experiments 
       (actually computing tag values) on the test set.  
    '''
    
    #unbox arguments
    db_fpath, db_name, output_folder, cross_val_folder, est_name, \
            param_one, value_one, param_two, value_two = args
    
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
        
        #Generate 50 random tags not used by any user in validation or test
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
        save_lhood = False
        if est_name == 'lda':
            est = create_lda_estimator(annotations, value_one, 
                num_items, num_tags, value_two)
            save_lhood = True
        else:
            est = create_bayes_estimator(annotations, value_one, value_two)
        
        param_out_folder = os.path.join(output_folder, \
                'params-%s-%f_%s-%f' % \
                (param_one, value_one, param_two, value_two))

        os.mkdir(param_out_folder)
        run_exp(user_items_to_filter, user_validation_tags, user_test_tags,
                user_to_item, num_items, random_tags, est, param_out_folder,
                save_lhood)
                
@plac.annotations(
    db_fpath = plac.Annotation('H5 database file', type=str),
    db_name = plac.Annotation('H5 database name', type=str),
    cross_val_folder = plac.Annotation('Folder with cross validation files', 
            type=str),
    output_folder = plac.Annotation('Output folder path', type=str),
    est_name = plac.Annotation('Estimator to perform grid search', type=str, 
            choices=['lda', 'smooth'], kind='option'),
    rand_seed = plac.Annotation('Random seed to use (None = default seed)',
            type=int, kind='option'),
    num_cores = plac.Annotation('Number of cores to use', type=int))
def main(db_fpath, db_name, cross_val_folder, output_folder, est_name, 
         rand_seed=None, num_cores=-1):
    '''Dispatches jobs in multiple cores'''
    
    seed(rand_seed)
    
    #Basic asserts for the folder
    assert os.path.isdir(output_folder)
    assert len(os.listdir(output_folder)) == 0
    
    if est_name == 'lda':
        params = LDA_GAMMA_PARAMS
    elif est_name == 'smooth':
        params = SMOOTH_PARAMS
    else:
        raise Exception('Unknown estinator name %s' % est_name)
    
    if num_cores <= 0:
        num_cores = multiprocessing.cpu_count()
        
    pool = multiprocessing.Pool(num_cores)
    
    def params_generator():
        '''Generates arguments for each core to use'''
        for param_one, param_two in combinations(sorted(params.keys()), 2):
            values_one = params[param_one]
            values_two = params[param_two]
            
            for i in range(len(values_one)):
                for j in range(len(values_two)):
                    val_one = values_one[i]
                    val_two = values_two[j]

                    yield db_fpath, db_name, output_folder, cross_val_folder, \
                        est_name, param_one, val_one, param_two, val_two
    
    pool.map(run_one, params_generator()) #Run in parallel, go go cores!
    pool.close()
    pool.join()
    
if __name__ == '__main__':
    sys.exit(plac.call(main))
