# -*- coding: utf8

import numpy as np
cimport numpy as np

cpdef double entropy(np.ndarray[np.float_t, ndim=1] probabilities_x)

cpdef double mutual_information(
        np.ndarray[np.float_t, ndim=1] probabilities_x, 
        np.ndarray[np.float_t, ndim=1] probabilities_xy)
                         
cpdef double norm_mutual_information(
        np.ndarray[np.float_t, ndim=1] probabilities_x, 
        np.ndarray[np.float_t, ndim=1] probabilities_xy)
                              
cpdef double kullback_leiber_divergence(
        np.ndarray[np.float_t, ndim=1] probabilities_p, 
        np.ndarray[np.float_t, ndim=1] probabilities_q)