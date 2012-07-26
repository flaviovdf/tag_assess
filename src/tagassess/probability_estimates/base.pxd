cimport numpy as np

cdef class ProbabilityEstimator:

    cdef double prob_item(self, int item)

    cdef double prob_tag(self, int tag)
    
    cdef double prob_tag_given_item(self, int item, int tag)
    
    cdef double prob_user(self, int user)
    
    cdef double prob_user_given_item(self, int item, int user)

    cdef np.ndarray[np.float_t, ndim=1] vect_prob_user(self, 
            np.ndarray[np.int_t, ndim=1] users)
            
    cdef np.ndarray[np.float_t, ndim=1] vect_prob_item(self, 
            np.ndarray[np.int_t, ndim=1] items)

    cdef np.ndarray[np.float_t, ndim=1] vect_prob_tag(self, 
            np.ndarray[np.int_t, ndim=1] tags)

    cdef np.ndarray[np.float_t, ndim=1] vect_prob_user_given_item(self,
            np.ndarray[np.int_t, ndim=1] items, int user)
        
    cdef np.ndarray[np.float_t, ndim=1] vect_prob_tag_given_item(self,
            np.ndarray[np.int_t, ndim=1] items, int tag)    