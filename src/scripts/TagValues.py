# -*- coding: utf8
'''
Simple scripts which prints the value of tags for a given user.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from tagassess import value_calculator
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator

#Regular Imports
from tagassess.dao.mongodb.annotations import AnnotReader

import argparse
import traceback

import sys

def real_main(database, table, smooth_func, lambda_, user):
    with AnnotReader(database) as reader:
        reader.change_table(table)
        est = SmoothEstimator(smooth_func, lambda_, reader.iterate())
        vc = value_calculator.ValueCalculator(est)
    
        itag_value = vc.tag_value_ucontext(user)
        for tag, tag_val in itag_value.iteritems():
            print(tag, tag_val)

def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Computes tag values.')
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')
    
    parser.add_argument('smooth_func', choices=['JM', 'Bayes'],
                        type=str,
                        help='Smoothing function to use (JM or Bayes)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')
    
    parser.add_argument('user', type=int,
                        help='User to consider')
    
    return parser
    
def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.database, vals.table, 
                         vals.smooth_func, vals.lambda_,
                         vals.user)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))