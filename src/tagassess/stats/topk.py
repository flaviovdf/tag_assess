# -*- coding: utf8
'''
Methods for comparing Top-K lists. See [1]_

References
----------
.. [1] Comparing top k lists.
       Ronald Fagin, Ravi Kumar, D. Sivakumar
       SIAM J. Discrete Mathematics 17, 1 (2003). PP/ 134-160
'''

from __future__ import division, print_function

import itertools

def kendall_tau_distance(data1, data2, k=10, p=0):
    '''
    Calculates the Kendall-Tau *distance* between two different
    rankings. This distance captures the amount of operations needed
    to converting one ranking to the other. For more info check on [1]_.
    
    The normalization is done considering the worse case when the
    intersection between the two ranks is empty.
    
    Arguments
    ----------
    data1: list (or any ordered iterable)
        The first ranking
    data2: list (or any ordered iterable)
        The second ranking
    k: int (default = 10)
        The top elements to consider from each ranking
    p: double [0, 1] (defaults to 0)
        The penalty factor
        
    References
    ----------
    .. [1] Comparing top k lists.
       Ronald Fagin, Ravi Kumar, D. Sivakumar
       SIAM J. Discrete Mathematics 17, 1 (2003). PP/ 134-160
    '''
    #Populates dictionaries with: element -> position
    positions1 = dict(itertools.izip(data1, xrange(1, k + 1)))
    positions2 = dict(itertools.izip(data2, xrange(1, k + 1)))
    
    #As set
    set1 = set(positions1.keys())
    set2 = set(positions2.keys())
    
    intersect = set1.intersection(set2)
    diff1 = set1.difference(set2)
    diff2 = set2.difference(set1)
    
    #Both elements in the intersection
    intersect_penalty = 0
    for i, j in itertools.combinations(intersect, 2):
        geq1 = positions1[i] >= positions1[j]
        geq2 = positions2[i] >= positions2[j]
    
        #The xor captures the different order.
        if geq1 ^ geq2:
            intersect_penalty += 1
    
    #Sum of ranks in distinct elements
    sum1 = sum(positions1[i] for i in diff1)
    sum2 = sum(positions2[i] for i in diff2)
    
    z = len(intersect)
    p = 0
    unnorm = (k - z) * ((2 + p) * k - p * z + 1 - p) + \
             (intersect_penalty - sum1 - sum2)
    
    #Computes Kendall-Tau
    norm_factor = (k * k) * (p + 1) - k * p
    return unnorm / norm_factor