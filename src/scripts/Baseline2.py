# -*- coding: utf8
from __future__ import division, print_function

from tagassess import value_calculator
from tagassess import graph
from tagassess import smooth

import numpy as np
import sys
import time

def log(msg):
    t = time.asctime()
    print(t, msg)

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
     
    tag_nodes, item_nodes, graph_rep = \
     graph.create_igraph(base_index, tag_to_item_index, uniq=False)
    item_trans = dict((v, k) for k, v in item_nodes.items())
    log('Graph has %d tags, %d items and %d edges'%(len(tag_nodes), len(item_nodes), graph_rep.ecount()))
    
    log('Experiment!')
    i = 0
    for tup in sorted(item_values):
        item = tup[1]
        item_id_in_graph = item_trans[item]
        
        log('Getting all paths to item %d, which in the graph is %d'%(item, item_id_in_graph))
        log('Immediate neighbors ' + str(graph_rep.neighbors(item_id_in_graph, type=3)))
        all_paths = graph_rep.get_all_shortest_paths(item_id_in_graph, mode=2)
        
        log('Found %d paths'%len(all_paths))
        for path in all_paths:
            path_without_item = path[1:]
            vals = np.array([tag_values[i] for i in path_without_item])
            
            avg = np.mean(vals)
            print(path_without_item)
            print(len(path_without_item), avg, vals)
        
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