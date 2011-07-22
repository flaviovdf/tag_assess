# -*- coding: utf8
'''
Computes tag files and value files filtering out
half of the annotations per user.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

#Cython Imports
try:
    from cy_tagassess import value_calculator
    from cy_tagassess.probability_estimates import SmoothEstimator
except ImportError: #Fallback to python code
    print('!!! UNABLE TO IMPORT CYTHON MODULES ''')
    from tagassess import value_calculator
    from tagassess.probability_estimates import SmoothEstimator

#Regular Imports
from itertools import ifilter

from tagassess import index_creator
from tagassess import graph
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.recommenders import ProbabilityReccomender

import argparse
import collections
import io
import multiprocessing
import numpy as np
import os
import sys
import traceback
import tempfile

def create_graph(annotation_it, user_folder):
    ntags, nsinks, iedges = \
     graph.iedge_from_annotations(annotation_it, 1,
                                  False)
    tmp_fname = tempfile.mktemp()
    n_edges = 0
    with io.open(tmp_fname, 'w') as tmp:
        for source, dest in sorted(iedges):
            tmp.write(u'%d %d\n' % (source, dest))
            n_edges += 1

    with io.open(tmp_fname) as tmp:
        out_graph = os.path.join(user_folder, 'navi.graph')
        with io.open(out_graph, 'w') as out:
            out.write(u'#Nodes:  %d\n'%ntags)
            out.write(u'#Edges:  %d\n'%n_edges)
            out.write(u'#Directed\n')
            for line in tmp:
                out.write(line)

def compute_tag_values(est, tag_to_item, tag_pops, 
                       user, user_folder, items_to_consider):
    recc = ProbabilityReccomender(est)
    value_calc = value_calculator.ValueCalculator(est, recc)
    
    tag_value = value_calc.tag_value_ucontext(user, 
                                              gamma_items = items_to_consider)
    with io.open(os.path.join(user_folder, 'tag.values'), 'w') as values:
        values.write(u'#TAG POP D RHO D*RHO\n')
        for tag, tag_val in tag_value.iteritems():
            
            #Mean P(I|u)
            items = np.array([item for item in tag_to_item[tag]])
            mean_prob = value_calc.mean_prob_item_given_user(user, items)
            
            final_val = tag_val * mean_prob
            
            values.write(u'%d %d %.15f %.15f %.15f\n' % 
                         (tag, tag_pops[tag], tag_val, mean_prob, final_val))

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
        
        fname = 'user_%d' % user
        user_folder = os.path.join(out_folder, fname)
        os.mkdir(user_folder)
        
        #Initial information
        with io.open(os.path.join(user_folder, 'info'), 'w') as info:
            info.write(u'#UID: %d\n' %user)
            
            relevant_str = ' '.join([str(i) for i in relevant])
            annotated_str = ' '.join([str(i) for i in annotated])
            
            info.write(u'# %d relevant  items: %s\n' %(len(relevant), str(relevant_str)))
            info.write(u'# %d annotated items: %s\n' %(len(annotated), str(annotated_str)))
        
        #Create Graph
        create_graph(reader.iterate(query = query), user_folder)
    
        #Compute popularity
        tag_pop = collections.defaultdict(int)
        for annotation in reader.iterate(query = query):
            tag = annotation['tag']
            tag_pop[tag] += 1          
        
        #Items to consider - Gamma items
        items_to_consider = set(xrange(est.num_items()))
        items_to_consider.difference_update(set(annotated))
        
        #Compute tag value
        iterator = reader.iterate(query = query)
        tag_to_item, item_to_tag = \
            index_creator.create_double_occurrence_index(iterator, 
                                                        'tag', 'item')
        
        compute_tag_values(est, tag_to_item, tag_pop, user, user_folder, 
                           np.array([i for i in items_to_consider]))
        
        with io.open(os.path.join(user_folder, 'relevant_item.tags'), 'w') as rel:
            rel.write(u'#ITEM TAG\n')
            for item in relevant:
                for tag in item_to_tag[tag]:
                    rel.write(u'%d %d\n' %(item, tag))
                
def real_main(database, table, smooth_func, lambda_, user_profile_size,
              out_folder, n_proc):
    
    def generator():
        with AnnotReader(database) as reader:
            '''Yields parameters for each user'''
            reader.change_table(table)
            uitem_idx = index_creator.create_occurrence_index(reader.iterate(),
                                                              'user', 'item')
            
            filt = lambda u: len(uitem_idx[u]) >= 10
            for user in ifilter(filt, uitem_idx.iterkeys()):
                items = [item for item in uitem_idx[user]]
                half = len(items) // 2
                
                relevant = items[:half]
                annotated = items[half:]
                yield database, table, user, relevant, annotated, \
                      smooth_func, lambda_, user_profile_size, out_folder
        
    pool = multiprocessing.Pool(n_proc)
    pool.map(_helper, generator(), 50)
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
                         vals.user_profile_size, vals.out_folder,
                         vals.num_cores)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))