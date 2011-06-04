#!/usr/env python
# -*- coding: utf8
'''
Computes rank correlation from baseline files.
'''
from scipy import stats

import numpy as np
import sys

if len(sys.argv) != 2:
    print >>sys.stderr, 'Usage %s <in file>' %sys.argv[0]
    sys.exit(1)

in_fpath = sys.argv[1]

#Each row will have 4 cols: user, tag, value_method, value_baseline
data = np.genfromtxt(in_fpath)
users = np.unique(data[:,0])
print '#USER\tSPEAR-RHO\tSPEAR-P\tKD-RHO\tKD-P\tISZ?'
for user in users:
    #select based on first column
    udata = data[data[:,0] == user]

    #begin with spearman rank, no sort needed
    our_method = udata[:,2]
    baseline = udata[:,3]
    
    sp_rho, sp_p = stats.spearmanr(our_method, baseline)
    
    rank_our = udata[udata[:,2].argsort(),:]
    rank_base = udata[udata[:,3].argsort(),:]
    
    #The use of [::-1] is to inverse ours, since in our case more is better.
    kt_rho, kt_p = stats.kendalltau(rank_our[:,1][::-1], rank_base[:,1])
    
    print '%.1f\t%.5f\t%.5f\t%.5f\t%.5f\t%s'%(user, sp_rho, sp_p, kt_rho, kt_p, str(our_method.any()))