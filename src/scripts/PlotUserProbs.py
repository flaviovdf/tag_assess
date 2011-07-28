# -*- coding: utf8
'''
Plots P(i|u) and P(i|u,t)
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

#Cython Imports
try:
    from cy_tagassess import value_calculator
    from cy_tagassess.probability_estimates import SmoothEstimator
except ImportError: #Fallback to python code
    print('!!! UNABLE TO IMPORT CYTHON MODULES ''')
    from tagassess import value_calculator
    from tagassess.probability_estimates import SmoothEstimator

#Regular Imports
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.recommenders import ProbabilityReccomender

import argparse
import multiprocessing
import numpy as np
import os
import sys
import traceback

def write_points_file(data, fname):
    with open(fname, 'w') as f:
        for d in data:
            print(d, file=f)
    
def _helper(params):
    compute_for_user(*params)

def compute_for_user(database, table, user, smooth_func, lambda_, 
                     user_profile_size, out_folder):
    
    with AnnotReader(database) as reader:
        reader.change_table(table)
        
        #Probability estimator
        est = SmoothEstimator(smooth_func, lambda_, reader.iterate(),
                              user_profile_size = user_profile_size)
        recc = ProbabilityReccomender(est)
        value_calc = value_calculator.ValueCalculator(est, recc)
        
        fname = 'user_%d' % user
        user_folder = os.path.join(out_folder, fname)
        os.mkdir(user_folder)
        
        items = np.arange(est.num_items(), dtype=np.int64)
        
        dict_dkl = value_calc.tag_value_ucontext(user)
        
        v_piu = value_calc.prob_items_given_user(user, items)
        v_dkl = np.array([dict_dkl[tag] for tag in sorted(dict_dkl)])
        
        v_dkl_argsort = v_dkl.argsort()
        top_5_tags = v_dkl_argsort[:5]
        bottom_5_tags = v_dkl_argsort[len(v_dkl) - 5:]
        
        write_points_file(v_piu, os.path.join(user_folder, 'v_piu.dat'))
        write_points_file(v_dkl, os.path.join(user_folder, 'dkl.dat'))
        
        for i, tag in enumerate(top_5_tags):
            v_pitu = value_calc.prob_items_given_user_tag(user, tag, items)
            write_points_file(v_pitu, os.path.join(user_folder, 
                                                   'v_pitu_tag_%d_top_%d.dat' % (tag, i)))
        for i, tag in enumerate(bottom_5_tags):
            v_pitu = value_calc.prob_items_given_user_tag(user, tag, items)
            write_points_file(v_pitu, os.path.join(user_folder, 
                                                   'v_pitu_tag_%d_bottom_%d.dat' % (tag, 5 - i)))
        
def real_main(database, table, smooth_func, lambda_, user_profile_size,
              out_folder, n_proc, user_ids):
    
    def generator():
        with AnnotReader(database) as reader:
            '''Yields parameters for each user'''
            reader.change_table(table)
            for user in user_ids:
                yield database, table, user, \
                      smooth_func, lambda_, user_profile_size, out_folder
        
    pool = multiprocessing.Pool(n_proc)
    pool.map(_helper, generator())
    pool.close()
    pool.join()
            
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Filters databases for exp.')
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')

    parser.add_argument('smooth_func', choices=['JM', 'Bayes'],
                        type=str,
                        help='Smoothing function to use (JM or Bayes)')

    parser.add_argument('lambda_', type=float,
                        help='Lambda to use, between [0, 1]')

    parser.add_argument('user_profile_size', type=int,
                        help='Lambda to use, between [0, 1]')

    parser.add_argument('out_folder', type=str,
                        help='folder for filtered graphs')

    parser.add_argument('num_cores', type=int,
                        help='Number of cores to use')

    parser.add_argument('users', type=int, nargs='+',
                        help='Users to plot')
    
    return parser
    

def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.database, vals.table, 
                         vals.smooth_func, vals.lambda_, 
                         vals.user_profile_size, vals.out_folder,
                         vals.num_cores, vals.users)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))