# -*- coding: utf8

from __future__ import division, print_function

from tagassess import value_calculator
from tagassess import smooth

import argparse
import networkx as nx
import numpy as np
import sys
import traceback

def real_main(in_file, table, smooth_func, lambda_, graph_file):
    value_calc = value_calculator.ValueCalculator(in_file, table,
                                                  smooth_func, lambda_)
    value_calc.open_reader(False)
    num_tags = len(value_calc.est.tag_col_freq)
    
    navi_graph = nx.read_edgelist(graph_file, create_using = nx.DiGraph())
    pagerank_dict = nx.pagerank_scipy(navi_graph) 
    pageranks = [pagerank_dict[node_id] for node_id in sorted(pagerank_dict)]
    
    tag_pager = np.array(pageranks[:num_tags])
    item_pager = np.array(pageranks[num_tags:])
    
    del pagerank_dict
    del pageranks
    
    #Renormalization
    tag_pager /= tag_pager.sum()
    item_pager /= item_pager.sum()
    
    for tag_id, tag_pr in enumerate(tag_pager):
        t_prob = value_calc.get_tag_probability(tag_id)
        t_pop = value_calc.get_tag_popularity(tag_id)
        print('T', tag_id, tag_pr, t_prob, t_pop, sep='\t')

    for item_id, item_pr in enumerate(item_pager):
        i_prob = value_calc.get_item_probability(item_id)
        i_pop = value_calc.get_item_popularity(item_id)
        print('I', item_id, item_pr, i_prob, i_pop, sep='\t')

    num_items = len(item_pager)
    item_ids = np.arange(num_items)
    est = value_calc.est
    for tag_id in xrange(num_tags):
        ti_probs = est.vect_prob_tag_given_item(est, item_ids, tag_id)
        for item_id in item_ids:
            print('TI', tag_id, item_id, item_pr, ti_probs[item_id], sep='\t')
    
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Correlation with page rank')
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')
    
    parser.add_argument('smooth_func', choices=smooth.name_dict().keys(),
                        type=str,
                        help='Smoothing function to use (JM, Bayes or None)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')
    
    parser.add_argument('graph_file', type=str,
                        help='The graph file')
    
    return parser
    
def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        smooth_func = smooth.get_by_name(vals.smooth_func)
        return real_main(vals.in_file, vals.table, 
                         smooth_func, vals.lambda_,
                         vals.graph_file)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))