# -*- coding: utf8
# cython: cdivision = True
# cython: boundscheck = False
# cython: wraparound = False

'''Probability based on lda methods'''

from __future__ import division, print_function

from cython.parallel import prange

cimport base

import numpy as np
cimport numpy as np
np.import_array()

cdef double NAN = float('nan')

cpdef double prior(int joint_count, int global_count, int num_occurences, 
                   double parameter):
    '''
    This is an auxiliary method to compute a prior probability for LDA. All of 
    the dirichlet priors p(w | z), p(d | z) and  p(z | u) can be estimated 
    with this function by changing the parameters.
    
    For simplicity we shall discuss in terms of p(x | y). The final equation is:
    
    .. math :: p(x | y) = (N_{x,y} + \alpha / X) / (N_y + \alpha)
    
    Arguments
    ---------
    joint_count: int
        N_{x,y}, i.e, the number of times x occurs jointly with y
    global_count: int
        N_y, i.e, the number of time y occurs
    num_occurences: int
        X, i.e, the size of the X distribution space
    parameter: double
        the scaling parameter, commonly denoted by alpha, theta or gamma.
    '''
    if (global_count + parameter) == 0 or num_occurences == 0:
        return NAN

    cdef double numerator = (parameter / num_occurences) + joint_count
    cdef double denominator = global_count + parameter
    
    return numerator / denominator

cdef class LDAEstimator(base.ProbabilityEstimator):
    '''
    Implementation of the TTM2 approach as described in [1]_.This approach 
    uses LDA based estimation to compute:
    
        * p(z | u) -> probability of topic z given user u
        * p(d | z) -> probability of document d given topic z
        * p(w | z) -> probability of term w given topic z
    
    We decided to maintain the authors (user, document, term) annotation instead
    of (user, item, tag). This was done for code readability, but document and
    item are the same thing, as is tag and term.
    
    The conditional distribution being estimated by LDA is:
    
    ..math::  
        \mathbf{E}[P(z_i|\mathbf{w,z_{-1},d}) \propto \\
        \Theta_{w|z} * \Phi_{d|z} * \Psi_{z|u}
    
    where, \Theta, \Phi, \Psi correspond to p(w | z), p(d | z) and  p(z | u)
    respectively.
    
    References
    ----------
    ..[1] Harvey, M., Ruthven, I., & Carman, M. J. (2011). 
    "Improving social bookmark search using personalised latent variable 
    language models." 
    Proceedings of the fourth ACM international conference on Web search and 
    data mining - WSDM  ’11. doi:10.1145/1935826.1935898
    '''
    def __init__(self, annotation_it, int num_topics, double alpha, double
                 beta, double gamma, int num_iterations, int num_burn_in,
                 int sample_user_dist_every, int seed):
        super(LDAEstimator, self).__init__()
        
        if seed > 0:
            np.random.seed(seed)
        
        self.num_topics = num_topics
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        
        self.num_iterations = num_iterations
        self.num_burn_in = num_burn_in
        
        self.sample_user_dist_every = sample_user_dist_every
        
        self._gibbs_populate(annotation_it)
        self._gibbs_sample()

    cdef void _gibbs_populate(self, annotation_it):
        '''
        Reads the annotations from `annotation_it` iterator and populates
        count matrices with initial random assignments for gibbs sampling.
        Topics for each annotation triple (user, document, term) is selected
        uniformly from the number of topics. For all triples not in the 
        annotation iterator no topic is assigned, but it will be when actual
        gibbs sampling is performed. 
        '''
        
        cdef int user = 0
        cdef int document = 0
        cdef int term = 0
        cdef int topic = 0
        
        cdef dict user_cnt = {}
        cdef dict topic_cnt = {}
        cdef dict document_cnt = {}
        cdef dict term_cnt = {}
        
        #Naming convention here is to follow p(a|b) -> b_a, or since b is given
        #comes first
        cdef dict topic_term = {}
        cdef dict topic_document = {}
        cdef dict user_topic = {}
        cdef list annotations_list = []

        self.num_annotations = 0
        for annotation in annotation_it:
            user = <int> annotation['user']
            document = <int> annotation['item']
            term = <int> annotation['tag']
            
            #initial assignment is random
            topic = np.random.randint(0, high=self.num_topics)
            
            if user not in user_cnt:
                user_cnt[user] = 0

            if document not in document_cnt:
                document_cnt[document] = 0
            
            if topic not in topic_cnt:
                topic_cnt[topic] = 0
            
            if term not in term_cnt:
                term_cnt[term] = 0
            
            user_cnt[user] += 1
            topic_cnt[topic] += 1
            document_cnt[document] += 1
            term_cnt[term] += 1
            
            if (user, topic) not in user_topic:
                user_topic[user, topic] = 0
                
            if (topic, term) not in topic_term:
                topic_term[topic, term] = 0
                
            if (topic, document) not in topic_document:
                topic_document[topic, document] = 0
            
            user_topic[user, topic] += 1
            topic_term[topic, term] += 1
            topic_document[topic, document] += 1
            annotations_list.append((user, topic, document, term))
            
            self.num_annotations += 1
        
        self.num_users = len(user_cnt) 
        self.num_terms = len(term_cnt) 
        self.num_documents = len(document_cnt)
        
        #Populates count matrices
        self._populate_count_matrices(user_cnt, topic_cnt, document_cnt, 
                                      term_cnt, topic_term, topic_document,
                                      user_topic)
        
        #Populates annotations
        self._populate_annotations(annotations_list)
        return
    
    cdef void _populate_count_matrices(self, dict user_cnt, dict topic_cnt, 
                                       dict document_cnt, dict term_cnt,
                                       dict topic_term, dict topic_document,
                                       dict user_topic):
        
        self.user_topic_cnt = np.zeros((self.num_users, self.num_topics),
                                       dtype='i')
        self.topic_document_cnt = np.zeros((self.num_topics,self.num_documents),
                                           dtype='i')
        self.topic_term_cnt = np.zeros((self.num_topics, self.num_terms),
                                       dtype='i')

        self.user_topic_prb = np.zeros((self.num_users, self.num_topics),
                                       dtype='f')
        self.topic_document_prb = np.zeros((self.num_topics,self.num_documents),
                                           dtype='f')
        self.topic_term_prb = np.zeros((self.num_topics, self.num_terms),
                                       dtype='f')

        self.user_cnt = np.zeros(self.num_users, dtype='i')
        self.document_cnt = np.zeros(self.num_documents, dtype='i')
        self.topic_cnt = np.zeros(self.num_topics, dtype='i')

        for topic from 0 <= topic < self.num_topics:
            self.topic_cnt[topic] = topic_cnt[topic]
            
            for user from 0 <= user < self.num_users:
                if (user, topic) in user_topic:
                    self.user_topic_cnt[user, topic] = user_topic[user, topic]
                
                if topic == 0:
                    self.user_cnt[user] = user_cnt[user]
            
            for document from 0 <= document < self.num_documents:
                if (topic, document) in topic_document:
                    self.topic_document_cnt[topic, document] = \
                            topic_document[topic, document]

                if topic == 0:
                    self.document_cnt[document] = document_cnt[document]
            
            for term from 0 <= term < self.num_terms:
                if (topic, term) in topic_term:
                    self.topic_term_cnt[topic, term] = topic_term[topic, term]
        
        return
    
    cdef void _populate_annotations(self, list annotations_list):
        #Populates annotations
        self.annot_user = np.zeros(self.num_annotations, dtype='i')
        self.annot_topic = np.zeros(self.num_annotations, dtype='i')
        self.annot_document = np.zeros(self.num_annotations, dtype='i')
        self.annot_term  = np.zeros(self.num_annotations, dtype='i')
        
        cdef Py_ssize_t i
        for i from 0 <= i < self.num_annotations:
            user, topic, document, term = annotations_list[i]
            self.annot_user[i] = user
            self.annot_topic[i] = topic
            self.annot_document[i] = document
            self.annot_term[i] = term
        
        return
    
    cdef void _gibbs_sample(self):
        '''
        Performs actual gibbs sampling. Most of this implementation is based on
        [1]_ and [2]_. It was based on a pseudocode described in [3]_ and a 
        Java implementation found in [4]_.
        
        The original LDA is extended to consider the user dimension. From a 
        broad perspective this algorithm works as follows:
        
        while not converged:
            for each topic z in (user u, document d, term w)*
                sample a new topic based on 
                    ..math:: 
                        \mathbf{E}[P(z_i|\mathbf{w,z_{-1},d})] \propto \\
                        \Theta_{w|z} * \Phi_{d|z} * \Psi_{z|u}
                update topic count matrices
            
                after burn in
                    update Theta, Phi, Psi
        
        *this loop considers only valid annotations, i.e, triples which exist
        in the annotation iterator.
        
        The vector z_{-1} is simulated by removing the topic under consideration
        from the count matrices.
        
        References
        ----------
        [1] T. L. Griffiths and M. Steyvers, 
            "Finding scientific topics.," 
            Proceedings of the National Academy of Sciences of the United States 
            of America, vol. 101 Suppl , no. suppl_1, pp. 5228–35, Apr. 2004.
        [2] T. Griffiths,
            "Gibbs sampling in the generative model of latent dirichlet 
            allocation," Standford University, vol. 18, no. 11, p. 3, 2002.
        [3] http://cxwangyi.files.wordpress.com/2012/01/llt.pdf
        [4] http://arbylon.net/projects/LdaGibbsSampler.java
        '''
        
        cdef int user = 0
        cdef int old_topic = 0
        cdef int document = 0
        cdef int term = 0
        cdef int new_topic = 0
        cdef int useful_steps = 0

        self.user_topic_prb = np.zeros((self.num_users, self.num_topics),
                                       dtype='f')
        self.topic_document_prb = np.zeros((self.num_topics,self.num_documents),
                                           dtype='f')
        self.topic_term_prb = np.zeros((self.num_topics, self.num_terms),
                                       dtype='f')

        cdef Py_ssize_t iter        
        cdef Py_ssize_t annot
        
        cdef int sample_user = 1
        
        for iter from 0 <= iter < self.num_iterations:
            
            if ((iter + 1) % self.sample_user_dist_every) == 0:
                sample_user = 1
            else:
                sample_user = 0 
            
            for annot from 0 <= annot < self.num_annotations:
                user = self.annot_user[annot]
                old_topic = self.annot_topic[annot]
                document = self.annot_document[annot]
                term = self.annot_term[annot]
                
                #Update count matrices
                new_topic = self._gibbs_update(user, old_topic, document, term,
                                               sample_user)
                self.annot_topic[annot] = new_topic
                
                #After burn in we can begin considering probabilities
                if iter >= self.num_burn_in:
                    self._add_probabilities(user, new_topic, document, term)
                    useful_steps += 1
            
        #Average out the sums which were considered
        self._average_probs(useful_steps)
        return

    cdef void _add_probabilities(self, 
                                 int user, int topic, int document, int term):
        '''Increments the probability matrices, these will be averaged out
        when Gibbs sampling ends'''
    
        self.user_topic_prb[user, topic] += \
                self._est_prob_topic_given_user(user, topic)
            
        self.topic_document_prb[topic, document] += \
                self._est_prob_document_given_topic(topic, document)
                
        self.topic_term_prb[topic, term] += \
                self._est_prob_term_given_topic(topic, term)
                
        return

    cdef void _average_probs(self, int num_runs):
        '''Averages out probability matrices'''
        
        cdef int user = 0
        cdef int document = 0
        cdef int term = 0
        cdef int topic = 0
        
        for topic from 0 <= topic < self.num_topics:
            for user from 0 <= user < self.num_users:
                self.user_topic_prb[user, topic] /= num_runs
            
            for document from 0 <= document < self.num_documents:
                self.topic_document_prb[topic, document] /= num_runs
            
            for term from 0 <= term < self.num_terms:
                self.topic_term_prb[topic, term] /= num_runs
        
        return
        
    cpdef int _gibbs_update(self, int user, int old_topic, 
                            int document, int term, int sample_user):
        '''
        Performs the gibbs update step. Here if a topic exists for
        (user, tag, term) it will be removed and a new topic will be sampled.
        '''
        
        #decrease counts if topic was already assigned,
        #this is a shortcut to ignore this assignment in
        #the sampling (z_-1).
        self.user_topic_cnt[user, old_topic] -= 1
        self.topic_document_cnt[old_topic, document] -= 1
        self.topic_term_cnt[old_topic, term] -= 1
        self.topic_cnt[old_topic] -= 1
            
        #sample a new topic
        new_topic = self._sample_topic(user, document, term, sample_user)
        
        #update counts
        self.user_topic_cnt[user, new_topic] += 1
        self.topic_document_cnt[new_topic, document] += 1
        self.topic_term_cnt[new_topic, term] += 1
        self.topic_cnt[new_topic] += 1
    
        return new_topic
    
    cpdef int _sample_topic(self, int user, int document, int term,
                            int sample_user):
        '''
        Draws a random topic based on current count matrices. The topic
        is for a user, tagging a document with the given term. This sampling
        is based on the posterior.
        '''
        
        cdef np.ndarray[dtype=np.float_t, ndim=1] probs = \
                np.ndarray(self.num_topics) 
        cdef double sum_probs = 0
        
        #The probs array will have values proportional to the probabilities.
        cdef int topic
        for topic from 0 <= topic < self.num_topics:
            probs[topic] = self._est_posterior_prob(user, topic, document, 
                                                    term, sample_user)
            sum_probs += probs[topic]
        
        #Scaling probabilities to add up to one
        for topic from 0 <= topic < self.num_topics:
            probs[topic] = probs[topic] / sum_probs
            
        #Draw topic from a multinomial
        return np.random.multinomial(1, probs).argmax()

    cdef double _est_prob_topic_given_user(self, int user, int topic):
        '''Estimates p(topic z|user u) based on the topic count matrices'''
        return prior(self.user_topic_cnt[user, topic], 
                     self.user_cnt[user],
                     self.num_topics,
                     self.gamma)

    cdef double _est_prob_document_given_topic(self, int topic, int document):
        '''Estimates p(document d|topic z) based on the topic count matrices'''
    
        return prior(self.topic_document_cnt[topic, document], 
                     self.topic_cnt[topic],
                     self.num_documents,
                     self.alpha)

    cdef double _est_prob_term_given_topic(self, int topic, int term):
        '''Estimates p(term w|topic z) based on the topic count matrices'''
        
        return prior(self.topic_term_cnt[topic, term], 
                     self.topic_cnt[topic],
                     self.num_terms,
                     self.beta)

    cdef double _est_posterior_prob(self, int user, int topic, int document,
                                     int term, int sample_user):
        '''Estimates the posterior probability based on topic count matrices'''
        
        cdef double p_topic_gv_user
        cdef double p_document_gv_topic
        cdef double p_term_gv_topic
        
        p_topic_gv_user = self._est_prob_topic_given_user(user, topic)
        p_document_gv_topic = self._est_prob_document_given_topic(topic, 
                                                                  document)
        p_term_gv_topic = self._est_prob_term_given_topic(topic, term)
        
        if sample_user == 1:
            return p_topic_gv_user * p_document_gv_topic * p_term_gv_topic
        else:
            return p_document_gv_topic * p_term_gv_topic

    #Methods to be used after sampling
    def prob_topic_given_user(self, int user, int topic):
        '''
        Returns p(topic z|user u) based on the \Psi matrix. This is different
        from using the topic count matrices since \Psi is only updated after
        burn in, so it is more accurate.
        
        This method should be used after gibbs sampling is performed.
        '''
        return self.user_topic_prb[user, topic]

    def prob_document_given_topic(self, int topic, int document):
        '''
        Returns p(document d|topic z) based on the \Phi matrix. 
        This is different from using the topic count matrices since \Phi 
        is only updated after burn in. 
        
        This method should be used after gibbs sampling is performed.
        '''
        return self.topic_document_prb[topic, document]

    def prob_term_given_topic(self, int topic, int term):
        '''
        Returns p(term w|topic z) based on the \Theta matrix. 
        This is different from using the topic count matrices since \Theta 
        is only updated after burn in. 
        
        This method should be used after gibbs sampling is performed.
        '''
            
        return self.topic_term_prb[topic, term]

    def posterior_prob(self, int user, int topic, int document, int term):
        '''
        Returns the posterior probability:
        
        .. math::
            \mathbf{E}[P(z_i|\mathbf{w,z_{-1},d}) \propto \\
                        \Theta_{w|z} * \Phi_{d|z} * \Psi_{z|u}
                        
        Note that this is will not add up to one for all topics. It needs to be
        renormalized.
        '''
        return self.prob_topic_given_user(user, topic) * \
                self.prob_document_given_topic(topic, document) * \
                self.prob_term_given_topic(topic, term)

    def _get_topic_counts(self):
        return np.asarray(self.topic_cnt)

    def _get_user_counts(self):
        return np.asarray(self.user_cnt)

    def _get_document_counts(self):
        return np.asarray(self.document_cnt)

    def _get_user_topic_counts(self):
        return np.asarray(self.user_topic_cnt)

    def _get_topic_document_counts(self):
        return np.asarray(self.topic_document_cnt)

    def _get_topic_term_counts(self):
        return np.asarray(self.topic_term_cnt)

    def _get_user_topic_prb(self):
        return np.asarray(self.user_topic_prb)

    def _get_topic_document_prb(self):
        return np.asarray(self.topic_document_prb)

    def _get_topic_term_prb(self):
        return np.asarray(self.topic_term_prb)

    def _get_topic_assignments(self):
        
        cdef dict return_val = {}
        cdef Py_ssize_t i
        for i from 0 <= i < self.num_annotations:
            return_val[(self.annot_user[i], self.annot_document[i],
                        self.annot_term[i])] = self.annot_topic[i]
        
        return return_val

    #Methods used by the value calculator
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user(self, int user, 
             np.ndarray[np.int_t, ndim=1] gamma_items):
            
        '''
        Computes P(I|u), i.e., returns an array with the probability of each
        item given the user.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.

        In this estimator, we compute this probability as:
        
        .. math::
            p(i|u) & \propto & \sum_z p(i|z)p(z|u)
          
        Arguments
        ---------
        user: int
            User id
        gamma_items:
            Items to consider. 
        '''
            
        cdef Py_ssize_t num_items = gamma_items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] vp_iu = np.ndarray(num_items)
        
        cdef Py_ssize_t i
        cdef Py_ssize_t topic
        cdef double sum_probs = 0
        
        for i in prange(num_items, nogil=True, schedule='static'):
            vp_iu[i] = 0
            for topic from 0 <= topic < self.num_topics:
                vp_iu[i] += self.user_topic_prb[user, topic] * \
                             self.topic_document_prb[topic, gamma_items[i]]
                             
                 
            sum_probs += vp_iu[i]
        
        for i in prange(num_items, nogil=True, schedule='static'):
            vp_iu[i] /= sum_probs
            
        return vp_iu
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user_tag(self,
            int user, int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|t, u), i.e., returns an array with the probability of each
        item given the user and tag.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.

        In this estimator, we compute this probability as:
        
        .. math::
            p(i|t, u) & \propto & p(i|u) p(t|i, u)
          
        Arguments
        ---------
        user: int
            User id
        tag: int
            Tag id
        gamma_items:
            Items to consider. 
        '''
 
        cdef Py_ssize_t num_items = gamma_items.shape[0]
        cdef Py_ssize_t i
        cdef Py_ssize_t topic
        
        cdef np.ndarray[np.float_t, ndim=1] vpi_tu = np.ndarray(num_items)
        cdef double sum_probs = 0

        for i in prange(num_items, nogil=True, schedule='static'):
            vpi_tu[i] = 0

            for topic from 0 <= topic < self.num_topics:
                vpi_tu[i] += self.user_topic_prb[user, topic] * \
                        self.topic_document_prb[topic, gamma_items[i]] * \
                        self.topic_term_prb[topic, tag]

            sum_probs += vpi_tu[i]

        for i in prange(num_items, nogil=True, schedule='static'):
            vpi_tu[i] /= sum_probs

        return vpi_tu

    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_tag(self, 
            int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        '''
        Computes P(I|t), i.e., returns an array with the probability of each
        item given the tag.
        
        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.
        
        On this estimator p(i | t) is proportional to:
        
        ..math:: 
            p(i | t) \propto \sum_z p(t | z) p(i | z)  * p(z)
        
        p(w | z) and p(d | z) are already pre-computed. While p(z) is
        proportional to the number of times the topic occurs.
        
        Arguments
        ---------
        tag: int
            User id
        gamma_items:
            Items to consider. 
        '''
        
        cdef Py_ssize_t num_items = gamma_items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] vp_it = np.ndarray(num_items)
        
        cdef Py_ssize_t i
        cdef Py_ssize_t topic
        cdef double sum_probs = 0
        
        for i in prange(num_items, nogil=True, schedule='static'):
            vp_it[i] = 0
            for topic from 0 <= topic < self.num_topics:
                vp_it[i] += self.document_cnt[gamma_items[i]] * \
                             self.topic_document_prb[topic, gamma_items[i]] * \
                             self.topic_term_prb[topic, tag]
                 
            sum_probs += vp_it[i]
        
        for i in prange(num_items, nogil=True, schedule='static'):
            vp_it[i] /= sum_probs
            
        return vp_it
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items(self, 
           np.ndarray[np.int_t, ndim=1] gamma_items):
        
        '''
        Computes P(I), i.e., returns an array with the probability of each
        item.

        We note that this method considers that gamma_items are all of the
        items that exist, so the vector returned *will* be rescaled to sum to
        one.

        For this estimator this probability is proportional to the frequency
        of the item on the dataset. 
        
        ..math:: 
            p(i) = N_i / N

        Arguments
        ---------
        gamma_items:
            Items to consider.
        '''
        
        cdef Py_ssize_t num_items = gamma_items.shape[0]
        cdef np.ndarray[np.float_t, ndim=1] vp_i = np.ndarray(num_items)
        cdef double sum_probs = 0
        
        cdef Py_ssize_t i
        for i in prange(num_items, nogil=True, schedule='static'):
            vp_i[i] = self.document_cnt[gamma_items[i]]
            sum_probs += vp_i[i]
        
        for i in prange(num_items, nogil=True, schedule='static'):
            vp_i[i] = vp_i[i] / sum_probs

        return vp_i
