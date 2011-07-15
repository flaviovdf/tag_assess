# -*- coding: utf8
'''
Computes tag files and value files filtering out
half of the annotations per user.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from itertools import ifilter

from tagassess import index_creator
from tagassess import smooth
from tagassess import value_calculator
from tagassess import graph
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.probability_estimates import SmoothEstimator
from tagassess.recommenders import ProbabilityReccomender

import argparse
import collections
import os
import traceback
import tempfile
import sys

def create_graph(annotation_it, user, user_folder):
    ntags, nsinks, iedges = \
     graph.iedge_from_annotations(annotation_it, 1,
                                  False)
    n_nodes = ntags + nsinks
    
    tmp_fname = tempfile.mktemp()
    n_edges = 0
    with open(tmp_fname, 'w') as tmp:
        for source, dest in sorted(iedges):
            print(source, dest, file=tmp)
            n_edges += 1

    with open(tmp_fname) as tmp:
        out_graph = os.path.join(user_folder, 'navi.graph')
        with open(out_graph, 'w') as out:
            print('#Nodes:  %d'%n_nodes, file=out)
            print('#Edges:  %d'%n_edges, file=out)
            print('#Directed', file=out)
            for line in tmp:
                print(line[:-1], file=out)

def compute_tag_values(annotation_it, user, user_folder):
    smooth_func = smooth.bayes
    lambda_ = 0.25
    est = SmoothEstimator(smooth_func, lambda_, annotation_it)
    recc = ProbabilityReccomender(est)
    value_calc = value_calculator.ValueCalculator(est, recc)
    
    itag_value = value_calc.itag_value_ucontext(user)
    with open(os.path.join(user_folder, 'tag.values'), 'w') as values:
        for tag_val, tag in sorted(itag_value, reverse=True):
            print(tag, tag_val, file=values)
              
def real_main(database, table, out_folder):
    
    with AnnotReader(database) as reader:
        reader.change_table(table) 
        idx = index_creator.create_occurrence_index(reader.iterate(),
                                                     'user', 'item')
        
        filt = lambda u: len(idx[u]) >= 10
        for user in ifilter(filt, idx.iterkeys()):
            items = [item for item in idx[user]]
            half = len(items) // 2
            
            relevant = items[:half]
            annotated = items[half:]
            
            query = {'$or' : [
                              { 'user':{'$ne'  : user} }, 
                              { 'item':{'$nin' : annotated} }
                             ]
                    }
            
            fname = 'user_%d' % user
            user_folder = os.path.join(out_folder, fname)
            os.mkdir(user_folder)
            
            #Initial information
            with open(os.path.join(user_folder, 'info'), 'w') as info:
                print('#UID: %d' %user, file=info)
                print('#Relevant  items: %s' %str(relevant), file=info)
                print('#Annotated items: %s' %str(annotated), file=info)
            
            #Create Graph
            create_graph(reader.iterate(query = query), user, user_folder)
          
            #Compute tag value
            compute_tag_values(reader.iterate(query = query), user, user_folder)
            
            #Compute popularity
            tag_pop = collections.defaultdict()
            for annotation in reader.iterate(query = query):
                tag = annotation['tag']
                tag_pop[tag] += 1
                
            with open(os.path.join(user_folder, 'tag.pop'), 'w') as pops:
                for tag in tag_pop:
                    print(tag, tag_pop, file=pops)
                     
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Filters databases for exp.')
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')
    
    parser.add_argument('out_folder', type=str,
                        help='folder for filtered graphs')
    
    return parser
    

def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.database, vals.table, vals.out_folder)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))