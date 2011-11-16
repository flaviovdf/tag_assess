# -*- coding: utf8
'''
This script plot the value of each individual tags when considering a user
profile based on a zip-f distribution. We will compute:

    * p(i|s) from a zip-f based random number generator
    * p(i|t,s) = p(i|s) * (p(t|i) / p(i)) - Appendix A of draft

With this, we can compute the Kullback-Leibler divergence between these
two distributions.

Also, we compute two variants of the average value of items retrieved by a tag:

    * rho = mean(prob(I^t|s))
    * new_rho = mean(1.0 / -log2(prob(I^t|s)))
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
import sys
import traceback

def fetch_tags_and_items(reader, min_tag_freq=1):
    '''
    This method retrieves an array of every item id, another one for 
    every tag id and a dict mapping tag ids to the items ids annotated
    by every tag.
    
    Arguments
    ---------
    reader: `AnnotReader`
        reader which connects to DB
        
    min_tag_freq: int
        Indicates that we should ignore tags with a frequency lower than
        this argument.
    '''
    #TODO: We can do all this in one for, change if speed is an issue.
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
    
    return np.arange(len(items)), np.array(sorted(tags)), tag_to_item
                
def main(database, table, smooth_func, lambda_, alpha, min_tag_freq=1):

    with AnnotReader(database) as reader:
        reader.change_table(table)
        
        #Determine the items annotated by each tag and array of all items
        items_array, tags_array, tag_to_item = \
                fetch_tags_and_items(reader, min_tag_freq)
        n_items = items_array.shape[0]

        #Generates user profile based on zipf and computes value
        seeker_profile = np.array(np.random.zipf(alpha, n_items), 
                                  dtype='float64')
        seeker_profile /= seeker_profile.sum()
        
        #Value of each tag
        estimator = SmoothEstimator(smooth_func, lambda_, reader.iterate())
        prob_tags = estimator.vect_prob_tag(tags_array)
        for tag_id in tag_to_item:
            
            #Probabilities
            prob_tags_item = estimator.vect_prob_tag_given_item(items_array, 
                                                                tag_id)
            prob_item_seeker_tag = (prob_tags_item / prob_tags[tag_id]) * \
                    seeker_profile
            prob_item_seeker_tag /= prob_item_seeker_tag.sum() #Renormalize
            prob_items_tagged = seeker_profile[tag_to_item[tag_id]]
            
            #Metrics
            dkl = entropy.kullback_leiber_divergence(prob_item_seeker_tag, 
                                                     seeker_profile)
            rho = np.mean(prob_items_tagged)
            new_rho = np.mean(1.0 / -np.log2(prob_items_tagged)) 
            
            print(tag_id, rho, new_rho, dkl, rho * dkl, new_rho * dkl)

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
                    values.lambda_, values.alpha, values.min_tag_freq)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))