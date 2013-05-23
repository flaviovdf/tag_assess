#!/usr/env python
# -*- coding: utf8
'''
Computes rank correlation from baseline files.
'''
from __future__ import division, print_function

from tagassess.stats import topk
from scipy import stats

import numpy as np
import sys

def main(args=None):
    if not args: args = []
    
    if len(args) != 2:
        print('Usage %s <in file>' % sys.argv[0])
        return 1
    
    in_fpath = args[1]
    
    #Each row will have 10 cols: 
    #user, tag, value, baseline, mean_tag_prob, tag_freq, \
    #item_tag_prob, item_tag_prob, tag_item_freq, user_tags
    data = np.genfromtxt(in_fpath, skip_footer=1)
    users = np.unique(data[:,0])
    
    to_plot_corr = []
    for i, user in enumerate(users):
        #select based on first column
        udata = data[data[:,0] == user]
        
        our_method = udata[:,2]
        baseline = 1.0 / udata[:,3]
        
        #If the first is nan, user has too many tags leading to probs = 0
        if our_method[0] == np.nan:
            continue 
        
        rank_our  = udata[our_method.argsort(),:][:,1]
        rank_base = udata[baseline.argsort(),:][:,1]
        
        print(i, rank_our[:10])
        print(i, rank_base[:10])
        
        corr = topk.kendall_tau_distance(rank_our, rank_base)
        to_plot_corr.append(corr)
        
    corr_data = np.array(to_plot_corr)
    mean = np.mean(corr_data)
    std = np.std(corr_data)
    skew = stats.skew(corr_data)
    _10perc = stats.scoreatpercentile(corr_data, 10)
    _50perc = stats.scoreatpercentile(corr_data, 50)
    _90perc = stats.scoreatpercentile(corr_data, 90)
    
    print('Mean            %.3f' % mean)
    print('Std             %.3f' % std)
    print('Skew            %.3f' % skew)
    print('10%% Percentile  %.3f' % _10perc)
    print('50%% Percentile  %.3f' % _50perc)
    print('90%% Percentile  %.3f' % _90perc)
    print()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))