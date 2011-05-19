# -*- coding: utf8
'''
This is it! Finally our baseline implementation.
Hail to the king baby!
'''
from __future__ import division, print_function

from itertools import izip
from tagassess import value_calculator
from tagassess import graph
from tagassess import smooth

import numpy as np
import sys
import time

def log(msg, file_=sys.stderr):
    '''Logs to given file'''
    date = time.asctime()
    print('%s -- %s'%(date, msg), file=file_)

def get_shortest_paths_igraph(tag_nodes, sink_nodes, edges, user_items):
    '''Get's a dictionary with all shortest paths to items'''
    graph_rep = graph._create_igraph(tag_nodes, sink_nodes, edges)
    tag_nodes_list = range(len(tag_nodes))
    paths = graph_rep.shortest_paths(tag_nodes_list)
    
    return_val = {}
    for tag, sps in izip(tag_nodes_list, paths):
        return_val[tag] = {}
        for graph_id, old_id in sink_nodes.iteritems():
            dist = sps[graph_id]
            if dist != float('inf') and old_id in user_items:
                return_val[tag][old_id] = sps[graph_id]
            
    return return_val
        
def get_shortest_paths(tag_nodes, sink_nodes, edge_list, 
                       user_items, use_totem=False):
    if not use_totem:
        return get_shortest_paths_igraph(tag_nodes, sink_nodes, 
                                         edge_list, user_items)
    else:
        raise Exception('Not yet done!!')

def real_main(annotation_file, table, user, smooth_func, lambda_, 
              num_relevant):
    
    #Relevant items
    log('Getting relevant items')
    iitem_value = value_calculator.iitem_value(annotation_file, table,
                                               user, smooth_func, lambda_,
                                               False)
    
    user_items = set(i for v, i, u in iitem_value if u)
    items_for_val = [(v, i) for v, i, u in iitem_value if not u]
    
    #Tags
    log('Getting tag relevances')
    itag_value = value_calculator.itag_value(annotation_file, table, 
                                             user, smooth_func, lambda_, 
                                             num_relevant, items_for_val)
    tag_values = dict((tag, val) for val, tag, b in itag_value)
    
    log('Creating Graph')
    base_index = graph.extract_indexes_from_file(annotation_file, table)
    tag_nodes, sink_nodes, edges = graph.edge_list(base_index, uniq=False)
    
    #Free up some mem
    del base_index
    
    log('Computing shortest paths')
    shortest_paths = get_shortest_paths(tag_nodes, sink_nodes, edges, user_items)
    
    #Computing the actual baseline
    log('Baseline')
    for tag, paths in shortest_paths.iteritems():
        if len(paths) > 0:
            print(tag_values[tag], np.mean(paths.values()))
    
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