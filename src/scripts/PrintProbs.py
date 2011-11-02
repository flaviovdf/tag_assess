#-*- coding: utf8
'''
For a given annotation table in the database, a smooth function and 
a lambda value; this script prints out:

for every t in tags
    for every i in items
        * p(t) t is the given `tag`
        * p(i) i is an item
        * p(t|i) for all i in I (items)
        * p(i|t) for all i in I (items)
        * the popularity of the tag
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
import random
import sys
import traceback

def compute_probabilites(value_calculator, tag, items):
    '''
    Compute the probabilities we want to print out:
        * p(t) t a tag
        * p(i) i is an item
        * p(t|i) for all i in I (items)
        * p(i|t) for all i in I (items)
        * the popularity of the tag
    '''
    estimator = value_calculator.est
    
    prob_tag = estimator.prob_tag(tag)
    vprob_item = value_calculator.rnorm_prob_items(items)
    vprob_tag_given_item = estimator.vect_prob_tag_given_item(items, tag)
    vprob_item_given_tag = value_calculator.rnorm_prob_items_given_tag(items)
    pop_tag = estimator.tag_pop(tag)

    return (prob_tag, vprob_item, vprob_tag_given_item, 
            vprob_item_given_tag, pop_tag)
    
def main(database, table, smooth_func, lambda_):
    with AnnotReader(database) as reader:
        reader.change_table(table)
        
        estimator = SmoothEstimator(smooth_func, lambda_, reader.iterate())
        calculator = ValueCalculator(estimator, None)
        
        items = range(estimator.num_items())
        random.shuffle(items)
        gamma_items = items[:100]
        
        print('#tag', 'item', 'p(t)', 'p(i)', 'p(t|i)','p(i|t)', 'pop_tag', 
              sep=',')
        for tag in xrange(estimator.num_tags()):
            return_value = compute_probabilites(calculator, tag, gamma_items)
            prob_tag = return_value[0]
            vprob_item = return_value[1]
            vprob_tag_given_item = return_value[2]
            vprob_item_given_tag = return_value[3]
            pop_tag = return_value[4]
                
            for item in xrange(len(vprob_item)):
                print(tag, item, prob_tag, vprob_item[item],
                      vprob_tag_given_item[item], vprob_item_given_tag[item], 
                      pop_tag, sep=',')
                
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