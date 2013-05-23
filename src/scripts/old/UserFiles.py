# -*- coding: utf8
'''
Computes tag files and value files filtering out
half of the annotations per user.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from tagassess import value_calculator
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator

#Regular Imports
from itertools import ifilter

from tagassess import index_creator
from tagassess.dao.mongodb.annotations import AnnotReader

import argparse
import io
import multiprocessing
import numpy as np
import os
import sys
import traceback

def compute_tag_values(est, value_calc, tag_to_item, 
                       user, user_folder, items_to_consider):
    tag_value = value_calc.tag_value_personalized(user, 
                                                gamma_items = items_to_consider)
    with io.open(os.path.join(user_folder, 'tag.values'), 'w') as values:
        values.write(u'#TAG POP TF IDF TDF_IDF D RHO D*RHO\n')
        for tag, tag_val in enumerate(tag_value):
            
            #Mean P(I|u)
            items_w_tag = \
                np.array([item for item in tag_to_item[tag]], dtype=np.int64)
            vp_iu = value_calc.rnorm_prob_items_given_user(user, None)
            mean_prob = vp_iu[items_w_tag].mean()
            
            final_val = tag_val * mean_prob
            
            values.write(u'%d %d %.15f %.15f %.15f\n' % 
                         (tag, est.tag_pop(tag), tag_val, mean_prob, final_val))

def _helper(params):
    compute_for_user(*params)

def compute_for_user(database, table, user, relevant, annotated, 
                     smooth_func, lambda_, user_profile_size, out_folder):
    with AnnotReader(database) as reader:
        reader.change_table(table)
        
        #Relevant items by user are left out with this query
        query = {'$or' : [
                          { 'user':{'$ne'  : user} }, 
                          { 'item':{'$nin' : relevant} }
                         ]
                }
        
        #Probability estimator
        est = SmoothEstimator(smooth_func, lambda_, 
                              reader.iterate(query = query),
                              user_profile_size = user_profile_size)
        value_calc = value_calculator.ValueCalculator(est)
        
        fname = 'user_%d' % user
        user_folder = os.path.join(out_folder, fname)
        os.mkdir(user_folder)
        
        #Initial information
        with io.open(os.path.join(user_folder, 'info'), 'w') as info:
            info.write(u'#UID: %d\n' %user)
            
            relevant_str = ' '.join([str(i) for i in relevant])
            annotated_str = ' '.join([str(i) for i in annotated])
            
            info.write(u'# %d relevant  items: %s\n' %(len(relevant), 
                                                       str(relevant_str)))
            info.write(u'# %d annotated items: %s\n' %(len(annotated), 
                                                       str(annotated_str)))
        
        #Create Graph
        iterator = reader.iterate(query = query)
        tag_to_item, item_to_tag = \
            index_creator.create_double_occurrence_index(iterator, 
                                                         'tag', 'item')
            
        #Items to consider <-> Gamma items
        items_to_consider = set(xrange(est.num_items()))
        annotated_set = set(annotated)
        items_to_consider.difference_update(annotated_set)
        
        compute_tag_values(est, value_calc, tag_to_item, user, 
                           user_folder, 
                           np.array([i for i in items_to_consider]))
        
        relevant_tags_fpath = os.path.join(user_folder, 'relevant_item.tags')
        with io.open(relevant_tags_fpath, 'w') as rel:
            rel.write(u'#ITEM TAG\n')
            for item in relevant:
                for tag in item_to_tag[item]:
                    rel.write(u'%d %d\n' %(item, tag))
                
def real_main(database, table, smooth_func, lambda_, user_profile_size,
              num_relevant, out_folder, n_proc):
    
    def generator():
        with AnnotReader(database) as reader:
            '''Yields parameters for each user'''
            reader.change_table(table)
            uitem_idx = index_creator.create_occurrence_index(reader.iterate(),
                                                              'user', 'item')
            
            filt = lambda u: len(uitem_idx[u]) >= 30
            for user in ifilter(filt, uitem_idx.iterkeys()):
                items = [item for item in uitem_idx[user]]
                
                cut = len(items) - num_relevant
                relevant = items[cut:]
                annotated = items[:cut]
                yield database, table, user, relevant, annotated, \
                      smooth_func, lambda_, user_profile_size, out_folder
        
    pool = multiprocessing.Pool(n_proc)
    pool.map(_helper, generator(), chunksize = 50)
    pool.close()
    pool.join()
            
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Filters databases for exp.')
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')

    parser.add_argument('smooth_func', choices=['JM', 'Bayes'],
                        type=str,
                        help='Smoothing function to use (JM or Bayes)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')

    parser.add_argument('user_profile_size', type=int,
                        help='Lambda to use, between [0, 1]')

    parser.add_argument('num_items_to_hide', type=int,
                        help='Number of items to hide')

    parser.add_argument('out_folder', type=str,
                        help='folder for filtered graphs')

    parser.add_argument('num_cores', type=int,
                        help='Number of cores to use')
    
    return parser
    

def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.database, vals.table, 
                         vals.smooth_func, vals.lambda_, 
                         vals.user_profile_size, vals.num_items_to_hide, 
                         vals.out_folder, vals.num_cores)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))