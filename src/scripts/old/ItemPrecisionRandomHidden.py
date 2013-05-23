#!/usr/bin/env python
# -*- encoding: utf-8
'''
This script will do as follows:
    1. Select the top k user based on number of annotations
    2. Filter out x% of the user items, creating a new annotation list
    3. Compute p(i|u)
    4. Saves two files one with p(i|u) and another with the hidden items
'''

import numpy as np
import os
import plac
import sys

from random import seed
from random import shuffle

from tagassess.index_creator import create_occurrence_index
from tagassess.index_creator import create_double_occurrence_index
from tagassess.probability_estimates.lda_estimator import LDAEstimator
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator
            
def get_user_item_pairs_to_filter(users_to_consider, annotations, 
        perc_items=.1):
    
    '''
    Gets user item pairs to filter. A percentage (`perc_items`) is filtered for
    each user. 

    The code to guarantees that we do not delete items from the trace 
    completely, that is, while removing items for users we guarantee that
    at we do not make an item be annotated by zero users. Thus, this code does 
    not guarantee that exactly `perc_items` will be removed per user.
    '''
    
    user_to_items = {}
    item_to_users = {}
    
    user_to_items, item_to_users = create_double_occurrence_index(
            annotations, 'user', 'item')
    
    user_item_pairs_to_filter = {}
    for user in user_to_items:

        #num items to remove for this user
        num_item = int(perc_items * len(user_to_items[user]))
        
        #Generate random candidates
        user_items = [item for item in user_to_items[user]]
        shuffle(user_items) #in place shuffle
        
        to_remove = []
        for item in user_items[:num_item]:
            if len(item_to_users[item]) > 1: #at least one user left
                item_to_users[item].remove(user)
                to_remove.append(item)

        user_item_pairs_to_filter[user] = to_remove
    
    return user_item_pairs_to_filter

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
    
    num_topics = 200
    alpha = 0.1 * num_docs
    beta = 0.1 * num_tags
    gamma = 25
    iterations = 300
    burn_in = 200
    lda_estimator = LDAEstimator(annotations_it, num_topics, alpha, beta, 
            gamma, iterations, burn_in)
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
    smooth_estimator = SmoothEstimator('Bayes', lambda_, annotations,
                                       user_profile_size=10)
    return smooth_estimator

@plac.annotations(
    library_thing_annotations_fpath = plac.Annotation('LT Trace File', 
            type=str),
    output_folder = plac.Annotation('Output folder path', type=str),
    num_users = plac.Annotation('Number of top users', type=int, kind='option'),
    perc_items = plac.Annotation('Percent of items to hide', type=int, 
            kind='option'),
    estimator = plac.Annotation('Estimator to use', type=str, 
            choices=['lda', 'smooth'], kind='option'),
    rand_seed = plac.Annotation('Random seed to use (None = default seed)',
            type=int, kind='option'))
def main(library_thing_annotations_fpath, output_folder, 
         num_users=20, perc_items=.1, estimator='lda', rand_seed=None):
    
    seed(rand_seed)
    
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

    user_item_pairs_to_filter = \
            get_user_item_pairs_to_filter(users_to_consider, 
                    base_annotations)

    #Create estimator
    filtered_annotations = FilteredAnnotations(user_item_pairs_to_filter)
    annotations = filtered_annotations.annotations(base_annotations)
    if estimator == 'smooth':
        est = create_smooth_estimator(annotations)
    elif estimator == 'lda':
        est = create_lda_estimator(annotations, len(item_ids), len(tag_ids))
    else:
        raise Exception('Unknown estimator, please choose from {lda, smooth}')

    #Run experiment!
    annotations = filtered_annotations.annotations(base_annotations)
    user_to_item = create_occurrence_index(annotations, 'user', 'item')
    
    for user in users_to_consider:
        gamma_items = [item for item in xrange(len(item_ids)) \
                                    if item not in user_to_item[item]]

        probs_i_given_u = est.prob_items_given_user(user, 
                np.asarray(gamma_items))

        piu_fpath = os.path.join(output_folder, 'probs-user-%d.dat' % user)
        np.savetxt(piu_fpath, probs_i_given_u)

        hidden_fpath = os.path.join(output_folder, 
                'hidden-items-for-user-%d.dat' % user)
        np.savetxt(hidden_fpath, user_item_pairs_to_filter[user])
        
        item_ids_fpath = os.path.join(output_folder,
                'gamma-item-ids-user-%d.dat' % user)
        np.savetxt(item_ids_fpath, gamma_items)

if __name__ == '__main__':
    sys.exit(plac.call(main))
