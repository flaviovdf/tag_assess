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
from tagassess.dao.annotations import AnnotReader

import argparse
import numpy as np
import sys
import time
import traceback

SMOOTHS = {'JM':smooth.jelinek_mercer,
           'Bayes':smooth.bayes,
           'None':smooth.none}

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

def real_main(in_file, table, smooth_func, lambda_):
    log('Determining items by each user')
    user_to_items = defaultdict(set)
    with AnnotReader(in_file) as reader:
        iterator = reader.iterate(table)
        for annotation in iterator:
            user = annotation.get_user()
            item = annotation.get_item()
            user_to_items[user].add(item)
    
    for user in user_to_items:
    log('Getting relevant items')
    vc = value_calculator.ValueCalculator(in_file, table, 
                                          smooth_func, lambda_)
    vc.open_reader()
    
    iitem_value = vc.iitem_value(user)
    items_tagged = set()
    vals_for_unsed = set()
    for item, val, used in iitem_value:
        items_tagged.add(item)
        if not used:
            vals_for_unsed.add((val, item))
    
    #Tags
    log('Getting tag relevances')
    itag_value = vc.itag_value(user, num_relevant)
    tag_values = dict((tag, val) for val, tag, b in itag_value)
    
    log('Creating Graph')
    base_index = graph.extract_indexes_from_file(in_file, table)
    tag_nodes, sink_nodes, edges = graph.edge_list(base_index, uniq=False)
    
    #Free up some mem
    del base_index
    
    log('Computing shortest paths')
    shortest_paths = get_shortest_paths(tag_nodes, sink_nodes, edges, items_tagged)
    
    #Computing the actual baseline
    log('Baseline')
    for tag, paths in shortest_paths.iteritems():
        if len(paths) > 0:
            print(tag_values[tag], np.mean(paths.values()))

def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Computes the baseline.')
    
    parser.add_argument('in_file', type=str,
                        help='annotation h5 file to read from')
    
    parser.add_argument('table', type=str,
                        help='database table from the file')
    
    parser.add_argument('smooth_func', choices=SMOOTHS.keys(),
                        type=str,
                        help='Smoothing function to use (JM, Bayes or None)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')
    
    return parser
    
def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        smooth_func = SMOOTHS[vals.smooth_func]
        return real_main(vals.in_file, vals.table, 
                         smooth_func, vals.lambda_)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))