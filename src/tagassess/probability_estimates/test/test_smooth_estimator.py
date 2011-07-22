# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=R0915
#pylint: disable-msg=W0212
from __future__ import division, print_function

from tagassess import data_parser
from tagassess import test

import numpy as np
import unittest

from tagassess.test import PyCyUnit

class TestSmoothEstimator(PyCyUnit):

    def get_module_to_test(self):
        from tagassess.probability_estimates import SmoothEstimator
        return SmoothEstimator

    def setUp(self):
        super(TestSmoothEstimator, self).setUp()
        self.annots = []

    def __init_test(self, fpath):
        parser = data_parser.Parser()
        with open(fpath) as in_f:
            for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                self.annots.append(annot)
                    
    def tearDown(self):
        self.annots = None
        
    def test_prob_item(self):
        SmoothEstimator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)
        
        #Item probabilities
        self.assertEquals(p.prob_item(0), 5 / 10)
        self.assertEquals(p.prob_item(1), 1 / 10)
        self.assertEquals(p.prob_item(2), 2 / 10)
        self.assertEquals(p.prob_item(3), 1 / 10)
        self.assertEquals(p.prob_item(4), 1 / 10)
        
        self.assertEquals(p.log_prob_item(0), np.log2(5 / 10))
        self.assertEquals(p.log_prob_item(1), np.log2(1 / 10))
        self.assertEquals(p.log_prob_item(2), np.log2(2 / 10))
        self.assertEquals(p.log_prob_item(3), np.log2(1 / 10))
        self.assertEquals(p.log_prob_item(4), np.log2(1 / 10))
        
        #Vect methods
        expected = np.array([0.5, 0.1, 0.2, 0.1, 0.2])
        estimated = p.vect_prob_item(np.array(range(5)))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2(np.array([0.5, 0.1, 0.2, 0.1, 0.2]))
        estimated = p.vect_log_prob_item(np.array(range(5)))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())
    
    def test_prob_tag(self):
        SmoothEstimator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)

        prob_t0 = sum(p.prob_tag_given_item(i, 0) * p.prob_item(i) for i in xrange(5))
        prob_t1 = sum(p.prob_tag_given_item(i, 1) * p.prob_item(i) for i in xrange(5))
        prob_t2 = sum(p.prob_tag_given_item(i, 2) * p.prob_item(i) for i in xrange(5))
        prob_t3 = sum(p.prob_tag_given_item(i, 3) * p.prob_item(i) for i in xrange(5))
        prob_t4 = sum(p.prob_tag_given_item(i, 4) * p.prob_item(i) for i in xrange(5))
        prob_t5 = sum(p.prob_tag_given_item(i, 5) * p.prob_item(i) for i in xrange(5))
        
        self.assertEquals(p.prob_tag(0), prob_t0)
        self.assertEquals(p.prob_tag(1), prob_t1)
        self.assertEquals(p.prob_tag(2), prob_t2)
        self.assertEquals(p.prob_tag(3), prob_t3)
        self.assertEquals(p.prob_tag(4), prob_t4)
        self.assertEquals(p.prob_tag(5), prob_t5)
        
        self.assertAlmostEquals(p.log_prob_tag(0), np.log2(prob_t0))
        self.assertAlmostEquals(p.log_prob_tag(1), np.log2(prob_t1))
        self.assertAlmostEquals(p.log_prob_tag(2), np.log2(prob_t2))
        self.assertAlmostEquals(p.log_prob_tag(3), np.log2(prob_t3))
        self.assertAlmostEquals(p.log_prob_tag(4), np.log2(prob_t4))
        self.assertAlmostEquals(p.log_prob_tag(5), np.log2(prob_t5))
        
        #Vect methods
        expected = np.array([prob_t0, prob_t1, prob_t2, prob_t3, prob_t4, prob_t5]).round(5)
        estimated = p.vect_prob_tag(np.array(range(6))).round(5)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2([prob_t0, prob_t1, prob_t2, prob_t3, prob_t4, prob_t5]).round(5)
        estimated = p.vect_log_prob_tag(np.array(range(6))).round(5)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

    def test_tag_given_item(self):
        SmoothEstimator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)
            
        #Tag given item
        from tagassess.smooth import jelinek_mercer
        prob_i0_t0 = jelinek_mercer(2, 5, 3, 10, lamb)
        prob_i1_t0 = jelinek_mercer(0, 5, 3, 10, lamb)
        prob_i2_t0 = jelinek_mercer(1, 2, 3, 10, lamb)
        prob_i3_t0 = jelinek_mercer(0, 5, 3, 10, lamb)
        prob_i4_t0 = jelinek_mercer(0, 5, 3, 10, lamb)
        
        self.assertEquals(p.prob_tag_given_item(0, 0), prob_i0_t0)
        self.assertEquals(p.prob_tag_given_item(1, 0), prob_i1_t0)
        self.assertEquals(p.prob_tag_given_item(2, 0), prob_i2_t0)
        self.assertEquals(p.prob_tag_given_item(3, 0), prob_i3_t0)
        self.assertEquals(p.prob_tag_given_item(4, 0), prob_i4_t0)
    
        self.assertEquals(p.log_prob_tag_given_item(0, 0), np.log2(prob_i0_t0))
        self.assertEquals(p.log_prob_tag_given_item(1, 0), np.log2(prob_i1_t0))
        self.assertEquals(p.log_prob_tag_given_item(2, 0), np.log2(prob_i2_t0))
        self.assertEquals(p.log_prob_tag_given_item(3, 0), np.log2(prob_i3_t0))
        self.assertEquals(p.log_prob_tag_given_item(4, 0), np.log2(prob_i4_t0))
    
        #Vect methods
        expected = np.array([prob_i0_t0, prob_i1_t0, prob_i2_t0, prob_i3_t0, prob_i4_t0])
        estimated = p.vect_prob_tag_given_item(np.array(range(5)), 0)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2([prob_i0_t0, prob_i1_t0, prob_i2_t0, prob_i3_t0, prob_i4_t0])
        estimated = p.vect_log_prob_tag_given_item(np.array(range(5)), 0)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())
    
    def test_prob_user(self):
        SmoothEstimator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)
        
        #User and user given item
        prob = p.prob_user(0)
        expected_prob_u = p.prob_tag(0) * p.prob_tag(1) * p.prob_tag(2) 
        self.assertEquals(prob, expected_prob_u)
        
        prob = p.log_prob_user(0)
        self.assertAlmostEquals(prob, np.log2(expected_prob_u))

        prob = p.prob_user_given_item(0, 0)
        expected_prob_ugi = p.prob_tag_given_item(0, 0) * p.prob_tag_given_item(0, 1) * p.prob_tag_given_item(0, 2) 
        self.assertEquals(prob, expected_prob_ugi)
        
        prob = p.log_prob_user_given_item(0, 0)
        self.assertAlmostEquals(prob, np.log2(expected_prob_ugi))
    
    def test_prob_user_profile_one(self):
        SmoothEstimator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots, user_profile_size=1)
        
        #User and user given item
        prob = p.prob_user(0)
        expected_prob_u = p.prob_tag(0) 
        self.assertEquals(prob, expected_prob_u)
        
        prob = p.log_prob_user(0)
        self.assertAlmostEquals(prob, np.log2(expected_prob_u))

        prob = p.prob_user_given_item(0, 0)
        expected_prob_ugi = p.prob_tag_given_item(0, 0)
        self.assertEquals(prob, expected_prob_ugi)
        
        prob = p.log_prob_user_given_item(0, 0)
        self.assertAlmostEquals(prob, np.log2(expected_prob_ugi))

    def test_prob_user_profile_two(self):
        SmoothEstimator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots, user_profile_size=2)
        
        #User and user given item
        prob = p.prob_user(0)
        expected_prob_u = p.prob_tag(0) * p.prob_tag(2)
        self.assertEquals(prob, expected_prob_u)
   
    def test_bayes(self):
        SmoothEstimator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lamb = 0.3
        p = SmoothEstimator(smooth_func, lamb, self.annots)
        
        from tagassess.smooth import bayes
        prob_i0_t0 = bayes(2, 5, 3, 10, lamb)
        prob_i0_t1 = bayes(1, 5, 3, 10, lamb)
        prob_i0_t2 = bayes(0, 5, 1, 10, lamb)
        prob_i0_t3 = bayes(1, 5, 1, 10, lamb)
        prob_i0_t4 = bayes(1, 5, 1, 10, lamb)
        prob_i0_t5 = bayes(0, 5, 1, 10, lamb)
        
        self.assertAlmostEquals(p.prob_tag_given_item(0, 0), prob_i0_t0)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 1), prob_i0_t1)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 2), prob_i0_t2)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 3), prob_i0_t3)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 4), prob_i0_t4)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 5), prob_i0_t5)
        
if __name__ == "__main__":
    unittest.main()
