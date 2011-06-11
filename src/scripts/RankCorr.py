#!/usr/env python
# -*- coding: utf8
'''
Computes rank correlation from baseline files.
'''
from __future__ import division, print_function

from scipy import stats

import numpy as np
import sys

if len(sys.argv) != 2:
    print('Usage %s <in file>' % sys.argv[0])
    sys.exit(1)

in_fpath = sys.argv[1]

#Each row will have 4 cols: user, tag, value_method, value_baseline
data = np.genfromtxt(in_fpath)
users = np.unique(data[:,0])
print('#USER\tSPEAR-RHO\tSPEAR-P\tKD-RHO\tKD-P\tISZ?')
for user in users:
    #select based on first column
    udata = data[data[:,0] == user]

    #begin with spearman rank, no sort needed
    our_method = udata[:,2]
    baseline = 1.0 / udata[:,3]
    
    sp_rho, sp_p = stats.spearmanr(our_method, baseline)
    
    rank_our = udata[our_method.argsort(),:]
    rank_base = udata[baseline.argsort(),:]
    kt_rho, kt_p = stats.kendalltau(rank_our[:,1], rank_base[:,1])
    
    print(user, end='\t')
    print('%.5f' % sp_rho, end='\t')
    print('%.5f' % sp_p, end='\t')
    print('%.5f' % kt_rho, end='\t')
    print('%.5f' % kt_p, end='\t')
    print(str(our_method.any()))