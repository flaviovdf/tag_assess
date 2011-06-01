# -*- coding: utf8
'''
Baseline. This is a quite long code, explanation bellow:

sps = load shortest paths

for each user:
    remove half of hers user x item annotations from trace
    compute tag values
    compare with shortest paths
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from collections import defaultdict

from tagassess import value_calculator
from tagassess import smooth
from tagassess.dao.annotations import AnnotReader

import argparse
import numpy as np
import random
import sys
import time
import traceback

def log(msg):
    '''Simple logging function'''
    
    date = time.asctime()
    print('%s -- %s'%(date, msg), file=sys.stderr)

def items_by_user(in_file, table):
    '''Determining items by each user'''
    
    user_to_items = defaultdict(set)
    with AnnotReader(in_file) as reader:
        iterator = reader.iterate(table)
        for annotation in iterator:
            user = annotation.get_user()
            item = annotation.get_item()
            user_to_items[user].add(item)
            
    return user_to_items

def filter_users(user_to_items, num_items = 10):
    '''Determining users with more than `num_items` items'''
    
    good_users = defaultdict(list)
    for user in user_to_items.keys():
        used = user_to_items[user]
        if len(used) >= num_items:
            good_users[user].extend(used)
            random.shuffle(good_users[user])
            
    return good_users

def load_tag_values(shortest_paths_file, good_users):
    '''Loading pre-computed tag values'''
    
    #We only need the paths to the previous items for the user.
    items = set()
    for user in good_users:
        items.update(good_users[user])
    
    sps = defaultdict(lambda: defaultdict(int))
    with open(shortest_paths_file) as sps_file:
        for line in sps_file:
            spl = line.split()
            
            tag = int(spl[0])
            item = int(spl[1])
            distance = float(spl[2])
            
            if item in items:
                sps[tag][item] = distance
    return sps

def create_value_calculator(in_file, table, smooth_func, lambda_):
    '''Creating value calculator'''
    
    value_calc = value_calculator.ValueCalculator(in_file, table, 
                                                  smooth_func, lambda_)
    return value_calc

def set_where(value_calc, user, items_to_disconsider):
    '''Filtering user some of her items'''
    
    user_item_annotations = {'user':[user], 'item':set()}
    for item in items_to_disconsider:
        user_item_annotations['item'].add(item)
    
    value_calc.set_filter_out(user_item_annotations)
    value_calc.open_reader()
    
def get_tag_values(value_calc, items_to_consider, tags_to_consider):
    '''Computing tag values'''
        
    itag_value = value_calc.itag_value_gcontext(items_to_consider,
                                                tags_to_consider)
    tag_to_vals = {}
    for val, tag in itag_value:
        tag_to_vals[tag] = val
    return tag_to_vals

def get_baseline_value(sps, tag):
    '''Getting baseline tag value'''
    
    vals = []
    for item in sps[tag]:
        vals.append(sps[tag][item])
    return np.mean(vals)

def real_main(in_file, table, smooth_func, lambda_, shortest_paths_file):
    log('Here we go! smooth = %s ; lambda = %f' % (str(smooth_func), lambda_))
    
    log('Determining which items were used by each user')
    user_to_items = items_by_user(in_file, table)
    good_users = filter_users(user_to_items)
    del user_to_items
    log('Left with: %d users' % len(good_users))
    
    log('Loading tag values from %s ' % shortest_paths_file)
    sps = load_tag_values(shortest_paths_file, good_users)
    
    log('Creating tag value calculator')
    value_calc = create_value_calculator(in_file, table, smooth_func, lambda_)
    
    for user in good_users:
        log('Starting experiment for user %d' % user)
        
        half = len(good_users[user]) // 2
        first_half = good_users[user][half:]
        second_half = good_users[user][:half]
        
        log('Leaving out %d items for user' % len(first_half))
        set_where(value_calc, user, first_half)
        
        log('Computing tag values with other %d items' % len(second_half))
        tag_vals = get_tag_values(value_calc, second_half, sps.keys())
        
        for tag in tag_vals:
            base_val = get_baseline_value(sps, tag)
            print(user, tag, tag_vals[tag], base_val)
                
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