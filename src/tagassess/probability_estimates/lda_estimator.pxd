#-*- coding: utf8
from __future__ import division, print_function
cimport base

import numpy as np
cimport numpy as np

#TODO: This next line is a workaround due to a cython bug. Remove when cython 
#      .19 comes out
#cdef public api double __PYX_NAN = float('nan')

cdef class LDAEstimator(base.ProbabilityEstimator):
    
    cdef int num_iterations
    cdef int num_burn_in
    
    cdef int num_terms
    cdef int num_users
    cdef int num_documents
    cdef int num_topics
    
    cdef double alpha
    cdef double beta
    cdef double gamma
    
    cdef Py_ssize_t curr_iter
    cdef double[:] log_likelihoods_train
    cdef double final_log_likelihood
    
    #Indicates when are we going to sample from p(z|u) in gibbs routine.
    #Mimicks the parameter $\pi_u$.
    cdef int sample_user_dist_every
    
    #Naming convention here is to follow p(a|b) -> b_a, or since b is given
    #comes first. These variables are cython memoryviews, think of them as 
    #matrices, you can do memview[a, b].
    #
    #The count matrices are used internally in the Gibbs sampling approach,
    #while the prb matrices contain the probabilities after sampling
    cdef int[:, ::1] user_topic_cnt
    cdef int[:, ::1] topic_document_cnt
    cdef int[:, ::1] topic_term_cnt
    
    cdef int[:] user_cnt
    cdef int[:] document_cnt
    cdef int[:] topic_cnt    
    
    #The following matrices are used only after the burn in. They will
    #store an average of the sampled posteriors for probability estimation
    cdef double[:, ::1] user_topic_prb
    cdef double[:, ::1] topic_document_prb
    cdef double[:, ::1] topic_term_prb
    
    #Annotations represented as four arrays (user, topic, document, term)
    cdef int num_annotations
    cdef int[:] annot_user
    cdef int[:] annot_topic
    cdef int[:] annot_document
    cdef int[:] annot_term
    
    #Populate methods
    cdef void _gibbs_populate(self, annotation_it)
    cdef void _populate_count_matrices(self, dict user_cnt, dict topic_cnt, 
                                       dict document_cnt, dict term_cnt,
                                       dict topic_term, dict topic_document,
                                       dict user_topic)
    cdef void _populate_annotations(self, list annotations_list)
   
    #Gibbs sample methods
    cdef void _gibbs_sample(self)
    cpdef int _gibbs_update(self, int user, int old_topic, int document, 
                            int term, int sample_user)
    cdef double _get_likelihood(self)
    cdef void _accumulate(self)
    cdef void _average_probs(self, int num_runs)
    cpdef int _sample_topic(self, int user, int document, int term, 
                            int sample_user)
       
    #Probability estimation methods
    cdef double _est_prob_topic_given_user(self, int user, int topic)
    cdef double _est_prob_document_given_topic(self, int topic, int document)
    cdef double _est_prob_term_given_topic(self, int topic, int term)
    cdef double _est_posterior_prob(self, int user, int topic, int document,
                                     int term, int sample_user)
