#!/usr/bin/env python
# -*- encoding: utf-8
'''
This script will do as follows:
    1. Select the top k user based on number of annotations
    2. Filter out x% of the user tags, creating a new annotation list
        2a. Tags are only filtered if another user has tagged something with it
    3. Select 100 random tags not used by the k users
    4. Compute for each user the value of the x% hidden tags and the random ones
'''
# TODO: Make this script useful for any of our tagging traces using the database
#       adapters.
# TODO: Make script more configurable. Options such as different parameters for
#       estimators and so forth

import numpy as np
import os
import plac
import sys

from random import shuffle

from tagassess.common import ContiguousID
from tagassess.index_creator import create_occurrence_index
from tagassess.index_creator import create_double_occurrence_index
from tagassess.probability_estimates.lda_estimator import LDAEstimator
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator
from tagassess.value_calculator import ValueCalculator

class FilteredAnnotations(object):
    '''
    Auxiliary class which mocks annotation database by iterating over the old
    one and filtering out annotations based on tags selected for removal.
    
    Arguments
    ---------
    user_tag_pairs : dict of user to tags
        User tag pairs to be filtered out
    '''

    def __init__(self, user_tag_pairs):
        self.user_tag_pairs = user_tag_pairs
        
    def annotations(self, annotations_it):
        '''
        Generates new annotations based on the original iterator to annotations
        
        Arguments
        ---------
        annotations_it : iterator to annotations
        '''
        
        for annotation in annotations_it:
            user = annotation['user']
            tag = annotation['tag']
            
            if user in self.user_tag_pairs and tag in self.user_tag_pairs[user]:
                continue
            
            yield annotation
            
def create_annotations(annotations_file):
    '''
    Creates annotations from a LibraryThing annotations file.
    
    Arguments
    ---------
    annotations_file : file
        Annotations file to parse
    '''
    user_ids = ContiguousID()
    item_ids = ContiguousID()
    tag_ids = ContiguousID()
    annotation_list = []

    with open(annotations_file) as annotations:

        for line in annotations:
            spl = line.split()

            user = spl[0]
            item = spl[1]
            tag = spl[3]

            uid = user_ids[user]
            iid = item_ids[item]
            tid = tag_ids[tag]

            annotation = {}
            annotation['user'] = uid
            annotation['item'] = iid
            annotation['tag'] = tid

            annotation_list.append(annotation)

    return annotation_list, user_ids, item_ids, tag_ids

def user_tag_pairs_to_filter(users_to_consider, annotations, perc_tags=.1, \
        num_random_tags=100):
    
    '''
    Gets use tag pairs to filter. Random tags are filtered if they are used
    by more than one user. This method also returns random tags to compute
    value for.
    '''
    
    user_to_tags, tags_to_user = create_double_occurrence_index(annotations, 
            'user', 'tag')
    
    #Generate candidate tags for removal, they have to be used by more than
    #one user.
    tags_to_remove = {}
    for user in users_to_consider:
        possible_tags = []
        for tag in user_to_tags[user]:
            if len(tags_to_user[tag]) > 1: #We only consider tags with >1 user
                possible_tags.append(tag)
        
        #num tags to remove for this user
        num_tags = int(perc_tags * len(user_to_tags[user]))
        
        #Generate random candidates
        candidate_tags = possible_tags[:num_tags]
        shuffle(candidate_tags) #In place
        
        tags_to_remove[user] = candidate_tags
    
    #Generate Random tags
    possible_tags = range(len(tags_to_user))
    shuffle(possible_tags)
    random_tags = []
    
    for tag in possible_tags:
        used_or_hidden = False
        
        for user in users_to_consider:
            if tag in user_to_tags[user]: #user_to_tags was built before filter
                used_or_hidden = True
                break
        
        if not used_or_hidden:
            random_tags.append(tag)
        
        if len(random_tags) == num_random_tags:
            break 
        
    return tags_to_remove, random_tags

def create_lda_estimator(annotations_it, num_docs, num_tags):
    '''
    Creates the lda estimator with the parameters described in [1]_
    
    References
    ----------
    ..[1] Harvey, M., Ruthven, I., & Carman, M. J. (2011). 
    "Improving social bookmark search using personalised latent variable 
    language models." 
    Proceedings of the fourth ACM international conference on Web search and 
    data mining - WSDM  ’11. doi:10.1145/1935826.1935898
    '''
    
    num_topics = 50
    alpha = 0.1 * num_docs
    beta = 0.1 * num_tags
    gamma = 25
    iterations = 300
    burn_in = 200
    lda_estimator = LDAEstimator(annotations_it, num_topics, alpha, beta, gamma, 
            iterations, burn_in)
    return lda_estimator

def create_smooth_estimator(annotations):
    '''
    Creates smooth estimator with the best Bayes parameter described in [1]_
    
    References
    ----------
    [1]_ Personalization of Tagging Systems, 
    Wang, Jun, Clements Maarten, Yang J., de Vries Arjen P., and 
    Reinders Marcel J. T. , 
    Information Processing and Management, Volume 46, Issue 1, p.58-70, (2010)
    '''
    
    lambda_ = 10^5
    smooth_estimator = SmoothEstimator('Bayes', lambda_, annotations)
    return smooth_estimator

def run_one_user(user, value_calculator, gamma_items, tags_hidden, random_tags,\
        output_folder):
    '''Computes values for one user'''
    
    val_hidd = value_calculator.tag_value_personalized(user, 
            np.asarray(gamma_items, dtype='int64'),
            np.asarray(tags_hidden, dtype='int64'))
    
    val_rand = value_calculator.tag_value_personalized(user, 
            np.asarray(gamma_items, dtype='int64'),
            np.asarray(random_tags, dtype='int64'))
    
    np.savetxt(os.path.join(output_folder, 'user-%d-hidden.tags' % user), 
            tags_hidden)
    np.savetxt(os.path.join(output_folder, 'user-%d-random.tags' % user), 
            random_tags)
    np.savetxt(os.path.join(output_folder, 'user-%d-hidden.vals' % user), 
            val_hidd)
    np.savetxt(os.path.join(output_folder, 'user-%d-random.vals' % user), 
            val_rand)
    
def main(library_thing_annotations_fpath, output_folder, 
         num_users=20, perc_tags=.1, estimator='smooth', num_proc=-1):
    
    #Basic asserts for the folder
    assert os.path.isdir(output_folder)
    assert len(os.listdir(output_folder)) == 0
    
    #Load LT file
    base_annotations, user_ids, item_ids, tag_ids = \
            create_annotations(library_thing_annotations_fpath)

    #Get most popular users
    user_pop = np.zeros(len(user_ids))
    for annot in base_annotations:
        user_pop[annot['user']] += 1
    users_to_consider = user_pop.argsort()[::-1][:num_users]

    #Get user tag pairs to filter and random tags
    user_to_hidden_tags, random_tags = \
            user_tag_pairs_to_filter(users_to_consider, base_annotations, 
                                     perc_tags)

    #Create estimator
    filtered_annotations = FilteredAnnotations(user_to_hidden_tags)
    annotations = filtered_annotations.annotations(base_annotations)
    if estimator == 'smooth':
        est = create_smooth_estimator(annotations)
    else:
        est = create_lda_estimator(annotations, len(item_ids), len(tag_ids))

    #This next line is needed to create a new generator
    annotations = filtered_annotations.annotations(base_annotations)
    value_calculator = ValueCalculator(est, annotations)
    
    #Run experiment!
    annotations = filtered_annotations.annotations(base_annotations)
    user_to_item = create_occurrence_index(annotations, 'user', 'item')
    for user in users_to_consider:
        gamma_items = [item for item in xrange(len(item_ids)) \
                                    if item not in user_to_item[item]]
        tags_hidden = user_to_hidden_tags[user]
        run_one_user(user, value_calculator, gamma_items, tags_hidden, \
                random_tags, output_folder)
    
if __name__ == '__main__':
    sys.exit(plac.call(main))