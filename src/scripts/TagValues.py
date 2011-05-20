# -*- coding: utf8
'''
Simple scripts which prints the value of tags for a given user.
'''

from __future__ import division, print_function

from tagassess import smooth
from tagassess import value_calculator

import argparse
import traceback

import sys

def real_main(in_file, table, smooth_func, lambda_, user, num_relevant):
    vc = value_calculator.ValueCalculator(in_file, table, 
                                          smooth_func, lambda_)
    vc.open_reader()
    
    itag_value = vc.itag_value(user, num_relevant, False)
    for tag_val, tag, used in sorted(itag_value, reverse=True):
        print(tag, tag_val, used)

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
    
    parser.add_argument('user', type=int,
                    help='User to consider')
    
    parser.add_argument('num_relevant', type=int,
                        help='Number of relevant item to consider. -1 is all')
    
    return parser
    
def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        smooth_func = smooth.get_by_name(vals.smooth_func)
        return real_main(vals.in_file, vals.table, 
                         smooth_func, vals.lambda_,
                         vals.user, vals.num_relevant)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))