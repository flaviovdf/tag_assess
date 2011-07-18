# -*- coding: utf8
'''
Simple scripts which prints the value of tags for a given user.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

import pyximport; pyximport.install()

from tagassess import smooth
from tagassess import value_calculator
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.probability_estimates import SmoothEstimator
from tagassess.recommenders import ProbabilityReccomender

import argparse
import traceback

import sys

def real_main(database, table, smooth_func, lambda_, user):
    with AnnotReader(database) as reader:
        reader.change_table(table)
        est = SmoothEstimator(smooth_func, lambda_, reader.iterate())
        recc = ProbabilityReccomender(est)
        vc = value_calculator.ValueCalculator(est, recc)
    
        itag_value = vc.itag_value_ucontext(user)
        for tag_val, tag in sorted(itag_value, reverse=True):
            print(tag, tag_val)

def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Computes tag values.')
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')
    
    parser.add_argument('smooth_func', choices=smooth.name_dict().keys(),
                        type=str,
                        help='Smoothing function to use (JM, Bayes or None)')

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
        smooth_func = smooth.get_by_name(vals.smooth_func)
        return real_main(vals.database, vals.table, 
                         smooth_func, vals.lambda_,
                         vals.user)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))