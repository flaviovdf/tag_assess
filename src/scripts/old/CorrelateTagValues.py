# -*- coding: utf8
'''
Given a folder with the value of tags in a file named `tag.values`, the 
probability of relevance of items in a file named `item.probs` and the tags
which annotated each item on file `item_tag.pairs`; this script will compute
the correlation of the value metrics with the characteristics of the tags.
'''
#TODO: iterate over every folder and compute average correlations with
#      confidence interval
#
#TODO: correlate DKL with RHO
#
#TODO: perform linear regression with a simple model with the characteristics
from __future__ import division, print_function

from scipy import stats

import argparse
import numpy as np
import os
import sys
import traceback

#Column identifiers
TAG_COL = 0
RHO_COL = 1
SUP_COL = 2
DKL_COL = 3
RHO_DKL_COL = 4
SUP_DKL_COL = 5
N_ITEMS_COL = 6
PROB_TAG_COL = 7
POP_TAG_COL = 8
MEAN_PTI_COL = 9

def corr(vector1, vector2, log1=False, log2=False):
    '''Computes linear correlation of the two vectors'''

    if log1:
        x = np.log(vector1)
    else:
        x = vector1
        
    if log2:
        y = np.log(vector2)
    else:
        y = vector2
    
    return stats.pearsonr(x, y)

def load_item_tag_pairs(item_tag_pairs_fpath):
    '''Loads a dict with item_ids as keys and list of tag_ids as values'''

    return_val = {}
    with open(item_tag_pairs_fpath) as item_tag_pairs_file:
        for line in item_tag_pairs_file:
            if '#' in line:
                continue
            
            spl = line.split()
            tag_id = int(spl[0])
            item_id = int(spl[1])
            
            return_val.setdefault(item_id, set()).add(tag_id)
            
    return return_val

def get_rows_based_on_item_to_tag(item_to_tag, items_to_consider, tag_values):
    '''
    Get's the value rows based on the tag_ids which annotate the given items.
    This is necessary because the row numbers in `tag_values` matrix may not
    corresponde with tag_id = row_number.
    '''

    #Gets rows for each tag_id
    tags_mem = set()
    tags_array = []
    for item_id in items_to_consider:
        if item_id not in item_to_tag:
            continue
        
        for tag_id in item_to_tag[item_id]:
            if tag_id not in tags_mem:
                tag_val_array = tag_values[tag_values[:,TAG_COL] == tag_id]
                
                assert len(tag_val_array == 1)
                assert tag_val_array[0][0] == tag_id
                
                tags_mem.add(tag_id)
                tags_array.append(tag_val_array[0])

    #Converts to numpy matrix    
    n_tags = len(tags_array)
    n_cols = len(tag_val_array[0])
    return_val = np.ndarray(shape=(n_tags, n_cols), dtype = 'f')
    for i, tag_row in enumerate(tags_array):
        return_val[i] = tag_row 
    
    return return_val

def main(input_folder):
    #File paths
    tag_values_fpath = os.path.join(input_folder, 'tag.values')
    item_tag_pairs_fpath = os.path.join(input_folder, 'item_tag.pairs')
    item_probs_fpath = os.path.join(input_folder, 'item.probs')
    
    tag_values = np.loadtxt(tag_values_fpath)
    item_to_tag = load_item_tag_pairs(item_tag_pairs_fpath)
    item_probs = np.loadtxt(item_probs_fpath)[:,1] #we only need the second col

    #Sort items by probability    
    items_sorted_by_prob = item_probs.argsort()
    top_50_items = items_sorted_by_prob[-50:]
    botton_50_items = items_sorted_by_prob[:50]
    
    #Get tags in top and bottom 50 items
    tags_top_50 = get_rows_based_on_item_to_tag(item_to_tag, top_50_items, 
                                                tag_values)
    tags_bottom_50 = get_rows_based_on_item_to_tag(item_to_tag, botton_50_items, 
                                                   tag_values)
    
    #Correlation
    corr_sup_dkl = corr(tag_values[:,DKL_COL], tag_values[:,SUP_COL])
    corr_val_pop = corr(tag_values[:,SUP_DKL_COL], tag_values[:,POP_TAG_COL], 
                        log2=True)
    corr_val_prob = corr(tag_values[:,SUP_DKL_COL], tag_values[:,PROB_TAG_COL], 
                         log2=True)
    corr_val_pti = corr(tag_values[:,SUP_DKL_COL], tag_values[:,MEAN_PTI_COL], 
                        log2=True)
    corr_val_nitem = corr(tag_values[:,SUP_DKL_COL], tag_values[:,N_ITEMS_COL], 
                          log2=True)
    
    #Hyp0
    mean_top_50 = np.mean(tags_top_50, axis=0)
    mean_bottom_50 = np.mean(tags_bottom_50, axis=0)
    
    print('corr(DKL, SUP) = %f; pval = %f' % corr_sup_dkl)
    print('corr(VAL, POP) = %f; pval = %f' % corr_val_pop)
    print('corr(VAL, PROB) = %f; pval = %f' % corr_val_prob)
    print('corr(VAL, PTI) = %f; pval = %f' % corr_val_pti)
    print('corr(VAL, NItems) = %f; pval = %f' % corr_val_nitem)
    
    print()
    print('mean top 50 dkl = %f' % mean_top_50[DKL_COL])
    print('mean top 50 sup = %f' % mean_top_50[SUP_COL])
    print('mean top 50 dkl/sup = %f' % mean_top_50[SUP_DKL_COL])
    print()
    print('mean bottom 50 dkl = %f' % mean_bottom_50[DKL_COL])
    print('mean bottom 50 sup = %f' % mean_bottom_50[SUP_COL])
    print('mean bottom 50 dkl/sup = %f' % mean_bottom_50[SUP_DKL_COL])
    
def create_parser(prog_name):
    '''Creates the parser with the command line options'''
    
    parser = argparse.ArgumentParser(prog_name, description=__doc__)
    parser.add_argument('input_folder', type=str,
                        help='Folder to save files to')
        
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    try:
        return main(values.input_folder)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))
