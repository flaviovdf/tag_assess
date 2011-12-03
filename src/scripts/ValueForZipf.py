# -*- coding: utf8
'''
This script saves the value of each individual tags when considering a user
profile based on a zip-f distribution. We will compute:

    * $p(i|s)$ from a zip-f based random number generator
    * $p(i|t,s) = p(i|s) * (p(t|i) / p(t))$ - Appendix A of draft

With this, we can compute the Kullback-Leibler divergence between these
two distributions.

Also, we compute two variants of the average value of items retrieved by a tag:

    * rho = mean(prob(i|s) for every i in I^t)
    * inverse surprisal = mean(1.0 / -log2(prob(i|s)) for every i in I^t)
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '16/11/2011'

from collections import defaultdict

from cy_tagassess import entropy
from cy_tagassess.probability_estimates import SmoothEstimator

from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.index_creator import create_occurrence_index

import argparse
import numpy as np
import os
import sys
import traceback

def fetch_tags_and_items(reader, min_tag_freq=1):
    '''
    This method retrieves an array of every item id, another one for 
    every tag id and a dict mapping tag ids to the items ids annotated
    by every tag. We also return the popularity of each tag.
    
    Arguments
    ---------
    reader: `AnnotReader`
        reader which connects to DB
        
    min_tag_freq: int
        Indicates that we should ignore tags with a frequency lower than
        this argument.
    '''
    tag_to_item = {}
    tags = []
    items = set()
    
    #Filter some very infrequent tags?
    tag_pop = defaultdict(int)
    for row in reader.iterate():
        items.add(row['item'])
        tag_pop[row['tag']] += 1 
            
    temp_index = create_occurrence_index(reader.iterate(), 'tag', 'item')
    for tag_id in temp_index:
        if min_tag_freq == -1 or tag_pop[tag_id] >= min_tag_freq:
            tags.append(tag_id)
            tag_to_item[tag_id] = np.array([i for i in temp_index[tag_id]])
            
    return np.arange(len(items)), np.array(sorted(tags), dtype='int64'), \
            tag_to_item, tag_pop
                

def tag_values(estimator, tags_array, items_array, tag_to_item, seeker_profile, 
               tag_pop, outfile):
    '''
    Saves the value of each individual tag.
    '''
    #Value for each tag
    print('#tag_id', 'rho', 'surprisal', 'dkl', 'dkl*rho', 'dkl/surprisal',
          'n_items', 'prob_tag', 'pop_tag', 'mean_pti', file=outfile)
    prob_tags = estimator.vect_prob_tag(tags_array)
    for i, tag_id in enumerate(tags_array):
        
        #Probabilities
        prob_tag_items = estimator.vect_prob_tag_given_item(items_array, 
                                                            tag_id)
        prob_tag = prob_tags[i]
        prob_item_seeker_tag = (prob_tag_items / prob_tag) * seeker_profile
        prob_item_seeker_tag /= prob_item_seeker_tag.sum() #Renormalize
        prob_items_tagged = seeker_profile[tag_to_item[tag_id]]
        
        #Metrics
        dkl = entropy.kullback_leiber_divergence(prob_item_seeker_tag, 
                                                 seeker_profile)
        rho = np.mean(prob_items_tagged)
        surprisal = np.mean(1.0 / -np.log2(prob_items_tagged))
        mean_pti = np.mean(prob_tag_items[tag_to_item[tag_id]])
        pop_tag = tag_pop[tag_id]
        print(tag_id, rho, surprisal, dkl, rho * dkl, dkl / surprisal,
              len(prob_items_tagged), prob_tag, pop_tag, mean_pti, file=outfile)


def main(database, table, smooth_func, lambda_, alpha, 
         output_folder, min_tag_freq=1):

    assert os.path.isdir(output_folder), '%s is not a directory' % output_folder
    tag_value_fpath = os.path.join(output_folder, 'tag.values')
    item_tag_fpath = os.path.join(output_folder, 'item_tag.pairs')
    item_probs_fpath = os.path.join(output_folder, 'item.probs')

    with AnnotReader(database) as reader:
        reader.change_table(table)
        
        #Determine the items annotated by each tag and array of all items
        items_array, tags_array, tag_to_item, tag_pop = \
                fetch_tags_and_items(reader, min_tag_freq)
        
        #Generates user profile based on zipf and computes value
        n_items = items_array.shape[0]
        seeker_profile = np.zeros(n_items, dtype='float64')
        n_dists = 10
        for i in xrange(n_dists):
            seeker_profile += np.random.zipf(alpha, n_items)
        
        #Average it out and transform to probabilities
        seeker_profile /= n_dists
        seeker_profile /= seeker_profile.sum()
        
        #Tag Value
        estimator = SmoothEstimator(smooth_func, lambda_, reader.iterate())
        with open(tag_value_fpath, 'w') as tag_value_file:
            tag_values(estimator, tags_array, items_array, tag_to_item, 
                       seeker_profile, tag_pop, tag_value_file)
        
        #Item tag pairs
        with open(item_tag_fpath, 'w') as item_tag_file:
            print('#tag_id', 'item_id', file=item_tag_file)
            for tag_id in tag_to_item:
                for item_id in tag_to_item[tag_id]:
                    print(tag_id, item_id, file=item_tag_file)

        with open(item_probs_fpath, 'w') as item_probs_file:
            print('#item_id', 'prob', file=item_probs_file)
            for item_id, prob in enumerate(seeker_profile):
                print(item_id, prob, file=item_probs_file)

def create_parser(prog_name):
    desc = __doc__
    parser = argparse.ArgumentParser(prog_name, description=desc)
    parser.add_argument('database', type=str,
                        help='database to read annotations from')
    
    parser.add_argument('table', type=str,
                        help='table with annotations')

    parser.add_argument('smooth_func', choices=['JM', 'Bayes'],
                        type=str,
                        help='Smoothing function to use (JM or Bayes)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')

    parser.add_argument('alpha', type=float,
                        help='Parameter for the zip-f function')

    parser.add_argument('output_folder', type=str,
                        help='Folder to save files to')
    
    parser.add_argument('--min_tag_freq', type=float, default=-1,
                        help='Ignore tags with frequency less than this value')
    
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.database, values.table, values.smooth_func,
                    values.lambda_, values.alpha, values.output_folder,
                    values.min_tag_freq)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))