# -*- coding: utf8
'''
Computes a tag tag graph and the global value for
those tags
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from tagassess import value_calculator
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator

#Regular imports
from tagassess import index_creator
from tagassess import graph
from tagassess.dao.mongodb.annotations import AnnotReader

import argparse
import collections
import io
import numpy as np
import os
import sys
import traceback
import tempfile

def create_graph(annotation_it, out_folder):
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
        out_graph = os.path.join(out_folder, 'navi.graph')
        with io.open(out_graph, 'w') as out:
            out.write(u'#Nodes:  %d\n'%ntags)
            out.write(u'#Edges:  %d\n'%n_edges)
            out.write(u'#Directed\n')
            for line in tmp:
                out.write(line)

def compute_tag_values(smooth_func, lambda_, annotation_it, 
                       tag_to_item, tag_pops, out_folder):
    est = SmoothEstimator(smooth_func, lambda_, annotation_it)
    value_calc = value_calculator.ValueCalculator(est)
    
    tag_value = value_calc.tag_value_item_search()
    with io.open(os.path.join(out_folder, 'tag.values'), 'w') as values:
        for tag, tag_val in enumerate(tag_value):
            items = np.array([item for item in tag_to_item[tag]])
            mean_prob = value_calc.rnorm_prob_items(items).mean()
            final_val = tag_val * mean_prob
            values.write(u'%d %d %.15f %.15f %.15f\n' % 
                         (tag, tag_pops[tag], tag_val, mean_prob, final_val))
              
def real_main(database, table, smooth_func, lambda_,
              out_folder):
    
    with AnnotReader(database) as reader:
        reader.change_table(table) 
        
        #Create Graph
        create_graph(reader.iterate(), out_folder)
      
        #Compute popularity
        tag_pop = collections.defaultdict(int)
        for annotation in reader.iterate():
            tag = annotation['tag']
            tag_pop[tag] += 1
            
        #Compute tag value
        tag_to_item, item_to_tag = \
            index_creator.create_double_occurrence_index(reader.iterate(), 
                                                        'tag', 'item')
        compute_tag_values(smooth_func, lambda_,
                           reader.iterate(), tag_to_item, tag_pop, out_folder)

        with io.open(os.path.join(out_folder, 'relevant_item.tags'), 'w') as rel:
            rel.write(u'#ITEM TAG\n')
            for item in item_to_tag:
                for tag in item_to_tag[tag]:
                    rel.write(u'%d %d\n' %(item, tag))  
                     
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
    
    parser.add_argument('out_folder', type=str,
                        help='folder for filtered graphs')

    return parser
    

def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.database, vals.table, 
                         vals.smooth_func, vals.lambda_,
                         vals.out_folder)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))