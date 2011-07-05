#!/usr/env python
# -*- coding: utf8
'''
Computes rank correlation from baseline files.
'''
from __future__ import division, print_function

from scipy import stats

import numpy as np
import numpy.ma as ma
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
    
    to_plot_corr = np.ndarray(shape=(len(users), 6), dtype='f')
    to_plot_corr.fill(np.nan) #Fill with nan, these will represent invalids
    for i, user in enumerate(users):
        #select based on first column
        udata = data[data[:,0] == user]
        
        our_method = udata[:,2]
        baseline = 1.0 / udata[:,3]
        #If the first is nan, user has too many tags leading to probs = 0
        if our_method[0] == np.nan:
            continue 
        
        #begin with spearman rank, no sort needed
        sp_rho, sp_p = stats.spearmanr(our_method, baseline)
        
        #Kendall Tau
#        rank_our = udata[our_method.argsort(),:]
#        rank_base = udata[baseline.argsort(),:]
#        kt_rho, kt_p = stats.kendalltau(rank_our[:,1], rank_base[:,1])
        
        #Correlation with factors
        tag_freq = udata[:,5]
        mean_tag_item_freq = udata[:,7]
        mtz = mean_tag_item_freq > 0
        sp_rho2, sp_p2 = stats.spearmanr(our_method[mtz], baseline[mtz])
        
        pr_rho_tfo, pr_p_tfo = stats.pearsonr(tag_freq, our_method)
        pr_rho_ifo, pr_p_ifo = stats.pearsonr(mean_tag_item_freq, our_method)
        
        pr_rho_tfb, pr_p_tfb = stats.pearsonr(tag_freq, baseline)
        pr_rho_ifb, pr_p_ifb = stats.pearsonr(mean_tag_item_freq, baseline)
        
        sig = 1
        to_plot_corr[(i, 0)] = sp_rho if sp_p <= sig else float('nan')
        to_plot_corr[(i, 1)] = pr_rho_tfo if pr_p_tfo <= sig else float('nan')
        to_plot_corr[(i, 2)] = pr_rho_ifo if pr_p_ifo <= sig else float('nan')
        to_plot_corr[(i, 3)] = pr_rho_tfb if pr_p_tfb <= sig else float('nan')
        to_plot_corr[(i, 4)] = pr_rho_ifb if pr_p_ifb <= sig else float('nan')
        to_plot_corr[(i, 5)] = sp_rho2 if sp_p2 <= sig else float('nan')
        
    #Mask 'nan', should not be plotted.
    m_to_plot_corr = ma.masked_invalid(to_plot_corr)
    
    names = {0:'Spearman',
             1:'Pearson (our x tfreq)',
             2:'Pearson (our x mean_tfreq_item)',
             3:'Pearson (base x tfreq)',
             4:'Pearson (base x mean_tfreq_item)',
             5:'BAHHAHAHA'}
    
    for corr in xrange(6):
        corr_data = m_to_plot_corr[:,corr]
        valid = corr_data[~corr_data.mask] #Get unmasked
        invalid = corr_data[corr_data.mask]
        name = names[corr]
        total = (len(valid) + len(invalid))
        perc = len(valid) / total
        print('-- Stats for %s. %.2f%% (%d of %d) had pval < 0.05) ' % (name, perc, len(valid), total))
        
        if len(valid) > 1:
            mean = np.mean(valid)
            std = np.std(valid)
            skew = stats.skew(valid)
            _10perc = stats.scoreatpercentile(valid, 10)
            _50perc = stats.scoreatpercentile(valid, 50)
            _90perc = stats.scoreatpercentile(valid, 90)
        else:
            mean = 0
            std = 0
            skew = 0
            _10perc = 0
            _50perc = 0
            _90perc = 0
        
        print('Mean            %.3f' % mean)
        print('Std             %.3f' % std)
        print('Skew            %.3f' % skew)
        print('10%% Percentile  %.3f' % _10perc)
        print('50%% Percentile  %.3f' % _50perc)
        print('90%% Percentile  %.3f' % _90perc)
        print()
    
if __name__ == '__main__':
    sys.exit(main(sys.argv))