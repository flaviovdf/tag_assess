# -*- coding: utf8
'''
This is it! Finally our baseline implementation.
Hail to the king baby!
'''
from __future__ import division, print_function

from collections import defaultdict
from itertools import izip
from tagassess import value_calculator
from tagassess import graph
from tagassess import smooth

import numpy as np
import random
import sys
import time

def log(msg):
    t = time.asctime()
    print(t, msg)

def get_shortest_paths_igraph(tag_nodes, sink_nodes, edge_list):
    graph_rep = graph._create_igraph(tag_nodes, sink_nodes.keys(), edge_list)
    paths = graph_rep.shortest_paths(tag_nodes)
    
    return_val = {}
    for tag, sps in izip(tag_nodes, paths):
        return_val[tag] = {}
        for destiny in sink_nodes:
            item_id = sink_nodes[destiny]
            return_val[tag][item_id] = sps[destiny]
            
    return graph_rep, return_val
        
def get_shortest_paths(tag_nodes, sink_nodes, edge_list, use_totem=False):
    if not use_totem:
        return get_shortest_paths_igraph(tag_nodes, sink_nodes, edge_list)
    else:
        raise Exception('Not yet done!!')

def real_main(annotation_file, table, user, smooth_func, lambda_, num_relevant):
    #Relevant items
    log('Getting relevant items')
    iitem_value = value_calculator.iitem_value(annotation_file, table, 
                                               user, smooth_func, lambda_)
    item_values = [(v, i) for v, i, u in iitem_value if u == False]
    
    #Tags
    log('Getting tag relevances')
    itag_value = value_calculator.itag_value(annotation_file, table, 
                                             user, smooth_func, lambda_, 
                                             num_relevant, item_values)
    tag_values = dict((tag, val) for val, tag, x in itag_value)
    
    log('Creating Graph')
    base_index, tag_to_item_index = \
     graph.extract_indexes_from_file(annotation_file, table)
        
    tag_nodes, sink_nodes, edge_list = \
     graph.edge_list(base_index, tag_to_item_index, uniq=False)
    
    log('Computing shortest paths')
    graph_rep, shortest_paths = \
     get_shortest_paths(tag_nodes, sink_nodes, edge_list)
    
    results = defaultdict(list)
    baselines = {}
    log('Experiment!')
    i = 0
    for tup in sorted(item_values):
        i += 1
        log('Item %d of %d'%(i, len(item_values)))
        item = tup[1]
        
        for j in xrange(100):
            rand = random.randint(0, len(tag_nodes) - 1)
            rand_tag = tag_nodes[rand]
            baseline = shortest_paths[rand_tag][item]
            baselines[item] = baseline
            
            log('Search starting from tag %d'%rand_tag)
            if baseline == float('inf'):
                log('No path! Skipping')
                continue
            else:
                log('I need at least %d steps'%baseline)
            
            found = False
            break_loop = False
            steps = 1
            query = rand_tag
            visited = set()
            visited.add(query)
            while not found and not break_loop:
                #TYPE-1 -> Outgoing edges
                neighbors = graph_rep.neighbors(query, type=1)
                if item in neighbors:
                    found = True
                else:
                    max_importance = float('-inf')
                    max_importance_tag = None
                    for neighbor in neighbors:
                        if neighbor not in sink_nodes:
                            importance = tag_values[neighbor]
                            if importance > max_importance:
                                max_importance = importance
                                max_importance_tag = neighbor
                                
                    query = max_importance_tag
                    steps += 1
                    
                if not query or query in visited:
                    log('No candidate found or returned to previous one. Breaking!')
                    break_loop = True
                else:
                    visited.add(query)
                    log('Going to tag %d'%query)
            
            if not break_loop:
                log('Yippe Kaye!! Found in %d steps'%steps)
                results[item].append(steps)
    
    for item in baselines:
        print(item, baselines[item], np.mean(results[item]))
        
def usage(prog_name, msg = None):
    '''Prints helps, msg if given and exits'''
    help_msg = 'Usage: %s <user> <smoothing type (jm | bayes)> <lambda> <num_relevant> <annotation_file> <table>'
    
    if msg:
        print(msg, file=sys.stderr)
    
    print(help_msg %prog_name, file=sys.stderr)
    return 1

def main(args=None):
    if not args: args = []
    
    if len(args) < 7:
        return usage(args[0])
        
    user = int(args[1])
    
    smoothing = args[2]
    if smoothing not in ('jm', 'bayes'):
        return usage(args[0], 'Unknown smoothing')
    
    smooth_func = None
    if smoothing == 'jm':
        smooth_func = smooth.jelinek_mercer
    else:
        smooth_func = smooth.bayes
    
    lambda_ = float(args[3])
    if (lambda_ <= 0 or lambda_ >= 1):
        return usage(args[0], 'Lambda should be in [0, 1]')
    
    num_relevant = int(args[4])
    annotation_file = args[5]
    table = args[6]

    real_main(annotation_file, table, user, smooth_func, lambda_, num_relevant)

if __name__ == '__main__':
    sys.exit(main(sys.argv))