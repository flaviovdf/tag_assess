# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=R0915
#pylint: disable-msg=W0212

from __future__ import division, print_function

from math import isnan

from tagassess import data_parser
from tagassess import test

from tagassess.probability_estimates.lda_estimator import LDAEstimator
from tagassess.probability_estimates.lda_estimator import prior

import numpy as np

import unittest

class TestLDAEstimator(unittest.TestCase):

    def create_annots(self, fpath):
        parser = data_parser.Parser()
        annots = []
        with open(fpath) as in_f:
            for annot in parser.iparse(in_f, 
                                       data_parser.delicious_flickr_parser):
                annots.append(annot)
        
        return annots

    def test_prior_computation(self):
        self.assertTrue(isnan(prior(1, 1, 0, 1)))
        self.assertTrue(isnan(prior(1, 0, 1, 0)))
        self.assertEqual(.631578947368421, prior(2, 3, 2, 0.8))

    def test_initial_population(self):
        annots = self.create_annots(test.SMALL_DEL_FILE)
        #With zero GIBBs will not run
        estimator = LDAEstimator(annots, 2, .5, .5, .5, 0, 0)
        
        user_cnt = estimator._get_user_counts()
        topic_cnt = estimator._get_topic_counts()
        document_cnt = estimator._get_document_counts()

        self.assertEquals(user_cnt[0], 4)
        self.assertEquals(user_cnt[1], 4)
        self.assertEquals(user_cnt[2], 2)
        
        self.assertEquals(document_cnt[0], 5)
        self.assertEquals(document_cnt[1], 1)
        self.assertEquals(document_cnt[2], 2)
        self.assertEquals(document_cnt[3], 1)
        self.assertEquals(document_cnt[4], 1)
        
        #10 assignments = 10 topics
        self.assertEquals(10, sum(topic_cnt))

        user_topic_cnt = estimator._get_user_topic_counts()
        topic_document_cnt = estimator._get_topic_document_counts()
        topic_term_cnt = estimator._get_topic_term_counts()

        #We can only test shapes and sum, since assignments are random
        self.assertEqual((3, 2), user_topic_cnt.shape)
        self.assertEqual((2, 5), topic_document_cnt.shape)
        self.assertEqual((2, 6), topic_term_cnt.shape)
        
        self.assertEqual(10, user_topic_cnt.sum())
        self.assertEqual(10, topic_document_cnt.sum())
        self.assertEqual(10, topic_term_cnt.sum())

        topic_assigments = estimator._get_topic_assignments()
        self.assertEqual(10, len(topic_assigments))
        self.assertEqual(10, estimator._get_topic_counts().sum())

        #Were the topics populated correctly?
        for annot in annots:
            aux = (annot['user'], annot['item'], annot['tag'])
            self.assertTrue(aux in topic_assigments)
            
        #Simple sanity check on topic assigmnets. Check if topics have valid
        #ids and if count matches count matrix        
        from collections import Counter
        c = Counter(topic_assigments.values())
        for topic in c:
            self.assertTrue(topic in [0, 1])
            self.assertTrue(c[topic] == topic_cnt[topic])

    def test_random_topic_selection(self):
        
        #Simply checks if probabilities come from valid space.
        
        annots = self.create_annots(test.DELICIOUS_FILE)
        estimator = LDAEstimator(annots, 2, .5, .5, .5, 0, 0) 
    
        for _ in xrange(1000):
            self.assertTrue(estimator._sample_topic(0, 0, 0) in [0, 1])
    
    def test_gibbs_update(self):
        
        #This test checks if topic assignment are decrased and re-increased
        
        annots = self.create_annots(test.DELICIOUS_FILE)
        estimator = LDAEstimator(annots, 2, .5, .5, .5, 0, 0)

        for annot, old_topic in estimator._get_topic_assignments().items():
            user, document, term = annot
            
            old_ut = estimator._get_user_topic_counts()[user, old_topic]
            old_td = estimator._get_topic_document_counts()[old_topic, document]
            old_tr = estimator._get_topic_term_counts()[old_topic, term]
            
            new_topic = estimator._gibbs_update(user, old_topic, document, term)
            new_ut = estimator._get_user_topic_counts()[user, old_topic]
            new_td = estimator._get_topic_document_counts()[old_topic, document]
            new_tr = estimator._get_topic_term_counts()[old_topic, term]
            
            if old_topic != new_topic:
                self.assertEqual(new_ut, old_ut - 1)
                self.assertEqual(new_td, old_td - 1)
                self.assertEqual(new_tr, old_tr - 1)
            else:
                self.assertEqual(new_ut, old_ut)
                self.assertEqual(new_td, old_td)
                self.assertEqual(new_tr, old_tr)                

        self.assertEqual(len(estimator._get_topic_assignments()), 
                         estimator._get_topic_counts().sum())

    def test_gibbs_sample(self):
        
        #Runs everything on a large dataset

        annots = self.create_annots(test.DELICIOUS_FILE)
        estimator = LDAEstimator(annots, 10, .5, .5, .5, 5, 2)
        
        ut = estimator._get_user_topic_prb()
        td = estimator._get_topic_document_prb()
        tt = estimator._get_topic_term_prb()
        
        self.assertTrue(ut.any())
        self.assertTrue(td.any())
        self.assertTrue(tt.any())
        
        self.assertTrue((ut >= 0).all())
        self.assertTrue((td >= 0).all())
        self.assertTrue((tt >= 0).all())

        self.assertTrue((ut <= 1).all())
        self.assertTrue((td <= 1).all())
        self.assertTrue((tt <= 1).all())

    def test_valid_probabilities(self):

        def isvalid(probs):
            return probs.sum() <= 1.00001 and probs.sum() >= 0.99999 and \
                (probs >= 0).all() and \
                (probs <= 1).all()

        #4 runs, 2 for burn 1
        annots = self.create_annots(test.SMALL_DEL_FILE)
        estimator = LDAEstimator(annots, 2, .1, .2, .3, 2, 0)
        
        gamma = np.arange(5)
        prob_items = estimator.prob_items(gamma)
        prob_items_tag = estimator.prob_items_given_tag(0, gamma)
        prob_items_user = estimator.prob_items_given_user(0, gamma)
        prob_items_user_tag = estimator.prob_items_given_user_tag(0, 0, gamma)
        
        self.assertTrue(isvalid(prob_items))
        self.assertTrue(isvalid(prob_items_tag))
        self.assertTrue(isvalid(prob_items_user))
        self.assertTrue(isvalid(prob_items_user_tag))

if __name__ == "__main__":
    unittest.main()
