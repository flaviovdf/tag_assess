#-*- coding: utf8
'''
For a given annotation table in the database, a smooth function and 
a lambda value; this script prints out:

for every t in tags
    * p(t) t is the given `tag`
    * the popularity of the tag
    for every i in items
        * p(i|t) for all i in I (items)
        * the popularity of the tag on the item
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '01/11/2011'

try:
    from cy_tagassess.probability_estimates import SmoothEstimator
    from cy_tagassess.value_calculator import ValueCalculator
except ImportError:
    print('!!! UNABLE TO IMPORT CYTHON MODULES ''')
    from tagassess.probability_estimates import SmoothEstimator
    from tagassess.value_calculator import ValueCalculator

from tagassess.dao.mongodb.annotations import AnnotReader

import argparse
import numpy as np
import sys
import traceback

def main(database, table, smooth_func, lambda_):
    with AnnotReader(database) as reader:
        reader.change_table(table)
        
        estimator = SmoothEstimator(smooth_func, lambda_, reader.iterate())
        calculator = ValueCalculator(estimator, None)
        
        gamma_items = np.arange(estimator.num_items())
        
        print('#tag_id', 'item', 'p(t)', 'p(i|t)', 'pop_tag', 
              'pop_tag_on_item', sep='|')
        for tag in xrange(estimator.num_tags()):
            
            prob_tag = estimator.prob_tag(tag)
            pop_tag = estimator.tag_pop(tag)
            v_prob_it = calculator.rnorm_prob_items_given_tag(tag, gamma_items)
            
            for item in gamma_items:
                pop_tag_on_item = estimator.item_tag_pop(item, tag)
                print(tag, item, prob_tag, v_prob_it[item], pop_tag, 
                      pop_tag_on_item, sep='|')
                
def create_parser(prog_name):
    desc = __doc__
    parser = argparse.ArgumentParser(prog_name, description=desc)
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')

    parser.add_argument('smooth_func', choices=['JM', 'Bayes'],
                        type=str,
                        help='Smoothing function to use (JM or Bayes)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')

    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.database, values.table, values.smooth_func, 
                    values.lambda_)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))