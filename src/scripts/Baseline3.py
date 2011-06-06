# -*- coding: utf8
'''
This baseline compares the average shortest path of a tag for all items,
with its global information gain value.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '06/06/2011'

from collections import defaultdict
from tagassess import smooth
from tagassess import value_calculator

import argparse
import traceback
import numpy as np
import sys

def load_tag_values(shortest_paths_file):
    '''Loading pre-computed tag values'''
    
    sps = defaultdict(list)
    items = set()
    with open(shortest_paths_file) as sps_file:
        for line in sps_file:
            spl = line.split()
            tag = int(spl[0])
            item = int(spl[1])
            distance = float(spl[2])
            sps[tag].append(distance)
            items.add(item)
    
    asps = {}
    for tag in sps:
        asps[tag] = np.mean(asps[tag])
    return asps, [i for i in items]

def real_main(in_file, table, smooth_func, lambda_, shortest_paths_file):
    asps, items = load_tag_values(shortest_paths_file)
    val_calc = value_calculator.ValueCalculator(in_file, table,
                                                smooth_func, lambda_)
    val_calc.open_reader()
    itag_value = val_calc.itag_value_gcontext(items, asps.keys())
    for tag_val, tag in itag_value:
        print(tag, tag_val, asps[tag])

def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Computes tag values.')
    
    parser.add_argument('in_file', type=str,
                        help='annotation h5 file to read from')
    
    parser.add_argument('table', type=str,
                        help='database table from the file')
    
    parser.add_argument('smooth_func', choices=smooth.name_dict().keys(),
                        type=str,
                        help='Smoothing function to use (JM, Bayes or None)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')
    
    parser.add_argument('shortest_paths_file', type=str,
                        help='A file with the shortest paths')
    
    return parser
    
def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        smooth_func = smooth.get_by_name(vals.smooth_func)
        return real_main(vals.in_file, vals.table, 
                         smooth_func, vals.lambda_,
                         vals.shortest_paths_file)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))