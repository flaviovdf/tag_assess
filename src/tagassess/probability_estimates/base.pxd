cimport numpy as np

cdef class ProbabilityEstimator:

    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user(self, 
            int user, np.ndarray[np.int_t, ndim=1] gamma_items)

    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user_tag(self,
            int user, int tag, np.ndarray[np.int_t, ndim=1] gamma_items)
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_tag(self, 
            int tag, np.ndarray[np.int_t, ndim=1] gamma_items)
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items(self, 
           np.ndarray[np.int_t, ndim=1] gamma_items)