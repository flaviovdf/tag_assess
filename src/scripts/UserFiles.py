# -*- coding: utf8
'''
Computes tag files and value files filtering out
half of the annotations per user.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

import pyximport; pyximport.install()

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
import io
import os
import sys
import traceback
import tempfile

def create_graph(annotation_it, user_folder):
    ntags, nsinks, iedges = \
     graph.iedge_from_annotations(annotation_it, 1,
                                  False)
    n_nodes = ntags + nsinks
    
    tmp_fname = tempfile.mktemp()
    n_edges = 0
    with io.open(tmp_fname, 'w') as tmp:
        for source, dest in sorted(iedges):
            tmp.write(u'%d %d\n' % (source, dest))
            n_edges += 1

    with io.open(tmp_fname) as tmp:
        out_graph = os.path.join(user_folder, 'navi.graph')
        with io.open(out_graph, 'w') as out:
            out.write(u'#Nodes:  %d\n'%n_nodes)
            out.write(u'#Edges:  %d\n'%n_edges)
            out.write(u'#Directed\n')
            for line in tmp:
                out.write(line)

def compute_tag_values(est, idx, user, user_folder):
    recc = ProbabilityReccomender(est)
    value_calc = value_calculator.ValueCalculator(est, recc)
    
    itag_value = value_calc.itag_value_ucontext(user)
    with io.open(os.path.join(user_folder, 'tag.values'), 'w') as values:
        for tag_val, tag in itag_value:
            mean_prob = est.vect_prob_item([item for item in idx[tag]]).mean()
            values.write(u'%d %.15f %.15f\n' % (tag, tag_val, mean_prob))
              
def real_main(database, table, out_folder):
    
    with AnnotReader(database) as reader:
        reader.change_table(table) 
        uitem_idx = index_creator.create_occurrence_index(reader.iterate(),
                                                          'user', 'item')
        
        filt = lambda u: len(uitem_idx[u]) >= 10
        for user in ifilter(filt, uitem_idx.iterkeys()):
            items = [item for item in uitem_idx[user]]
            half = len(items) // 2
            
            relevant = items[:half]
            annotated = items[half:]
            
            query = {'$or' : [
                              { 'user':{'$ne'  : user} }, 
                              { 'item':{'$nin' : annotated} }
                             ]
                    }
            
            #Probability estimator
            smooth_func = smooth.bayes
            lambda_ = 0.25
            est = SmoothEstimator(smooth_func, lambda_, 
                                  reader.iterate(query = query))
            
            if est.prob_user(user) == 0:
                continue
            
            fname = 'user_%d' % user
            user_folder = os.path.join(out_folder, fname)
            os.mkdir(user_folder)
            
            #Initial information
            with io.open(os.path.join(user_folder, 'info'), 'w') as info:
                info.write(u'#UID: %d\n' %user)
                info.write(u'#Relevant  items: %s\n' %str(relevant))
                info.write(u'#Annotated items: %s\n' %str(annotated))
            
            #Create Graph
            create_graph(reader.iterate(query = query), user_folder)
          
            #Compute tag value
            idx = index_creator.create_occurrence_index(reader.iterate(), 
                                                        'tag', 'item')
            compute_tag_values(est, idx, user, user_folder)
            
            #Compute popularity
            tag_pop = collections.defaultdict(int)
            for annotation in reader.iterate(query = query):
                tag = annotation['tag']
                tag_pop[tag] += 1
                
            with io.open(os.path.join(user_folder, 'tag.pop'), 'w') as pops:
                for tag in tag_pop:
                    pops.write(u'%d %d\n' % (tag, tag_pop[tag]))
                     
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