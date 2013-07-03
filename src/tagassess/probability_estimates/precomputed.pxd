# -*- coding: utf8
'''Probability based on pre-computed values'''

cimport base

import numpy as np
cimport numpy as np

cdef class PrecomputedEstimator(base.ProbabilityEstimator):
    
    cdef list users_fpaths
    cdef dict user_to_piu
    cdef dict user_to_pitu
    cdef dict user_to_tags
    cdef dict user_to_gamma
