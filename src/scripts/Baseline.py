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

def log(msg, file_=sys.stderr):
    '''Logs to given file'''
    date = time.asctime()
    print('%s -- %s'%(date, msg), file=file_)

def get_shortest_paths_igraph(tag_nodes, sink_nodes, edges):
    '''Get's a dictionary with all shortest paths to items'''
    graph_rep = graph._create_igraph(tag_nodes, sink_nodes, edges)
    tag_nodes_list = range(len(tag_nodes))
    paths = graph_rep.shortest_paths(tag_nodes_list)
    
    return_val = {}
    for tag, sps in izip(tag_nodes_list, paths):
        return_val[tag] = {}
        for graph_id, old_id in sink_nodes.iteritems():
            dist = sps[graph_id]
            
            if dist != float('inf'):
                return_val[tag][old_id] = sps[graph_id]
            
    return return_val
        
def get_shortest_paths(tag_nodes, sink_nodes, edge_list, 
                       use_totem=False):
    if not use_totem:
        return get_shortest_paths_igraph(tag_nodes, sink_nodes, edge_list)
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
    
    log('Creating Graph')
    base_index = graph.extract_indexes_from_file(in_file, table)
    tag_nodes, sink_nodes, edges = graph.edge_list(base_index, uniq=False)
    
    log('Computing shortest paths')
    shortest_paths = get_shortest_paths(tag_nodes, sink_nodes, edges)
    
    #Free mem
    del base_index
    del tag_nodes
    del sink_nodes
    del edges

    log('Creating value calculator')    
    vc = value_calculator.ValueCalculator(in_file, table, 
                                          smooth_func, lambda_)
    vc.open_reader()
    for user in user_to_items:
        user_items = user_to_items[user]
        num_uitems = len(user_items)
        half = num_uitems / 2
        
        filter_query = ['(USER != %d)'%user]
        items_to_use = []
        for i, item in enumerate(user_items):
            filter_query.append('(ITEM != %d)'%item)
            items_to_use.append(item)
            if i >= half:
                break
        
        log('Estimating tag relevance - user == %d'%user)
        itag_value = vc.itag_value(user, -1, False, items_to_use)
        tag_values = dict((tag, (val, used)) for val, tag, used in itag_value)
    
        #Computing the actual baseline
        log('Baseline - user == %d'%user)
        for tag, paths in shortest_paths.iteritems():
            if len(paths) > 0:
                val, used = tag_values[tag]
                print(user, used, val, np.mean(paths.values()))

def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Computes the baseline.')
    
    parser.add_argument('in_file', type=str,
                        help='annotation h5 file to read from')
    
    parser.add_argument('table', type=str,
                        help='database table from the file')
    
    parser.add_argument('smooth_func', choices=smooth.name_dict().keys(),
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
        smooth_func = smooth.get_by_name(vals.smooth_func)
        return real_main(vals.in_file, vals.table, 
                         smooth_func, vals.lambda_)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))