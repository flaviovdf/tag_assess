# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=R0915
#pylint: disable-msg=W0212
from __future__ import division, print_function

from tagassess import data_parser
from tagassess import smooth
from tagassess import test
from tagassess.probability_estimates import SmoothEstimator

import numpy as np
import unittest

class TestAll(unittest.TestCase):

    def setUp(self):
        self.annots = []

    def __init_test(self, fpath):
        parser = data_parser.Parser()
        with open(fpath) as in_f:
            for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                self.annots.append(annot)
                    
    def tearDown(self):
        self.annots = None
        
    def test_prob_item(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.jelinek_mercer
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
        estimated = p.vect_prob_item(range(5))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2(np.array([0.5, 0.1, 0.2, 0.1, 0.2]))
        estimated = p.vect_log_prob_item(range(5))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())
    
    def test_prob_tag(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.jelinek_mercer
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
        
        self.assertEquals(p.log_prob_tag(0), np.log2(prob_t0))
        self.assertEquals(p.log_prob_tag(1), np.log2(prob_t1))
        self.assertEquals(p.log_prob_tag(2), np.log2(prob_t2))
        self.assertEquals(p.log_prob_tag(3), np.log2(prob_t3))
        self.assertEquals(p.log_prob_tag(4), np.log2(prob_t4))
        self.assertEquals(p.log_prob_tag(5), np.log2(prob_t5))
        
        #Vect methods
        expected = np.array([prob_t0, prob_t1, prob_t2, prob_t3, prob_t4, prob_t5])
        estimated = p.vect_prob_tag(range(6))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2([prob_t0, prob_t1, prob_t2, prob_t3, prob_t4, prob_t5])
        estimated = p.vect_log_prob_tag(range(6))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

    def test_tag_given_item(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.jelinek_mercer
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)
            
        #Tag given item
        prob_i0_t0 = smooth_func(2, 5, 3, 10, lamb)[0]
        prob_i0_t1 = smooth_func(1, 5, 3, 10, lamb)[0]
        prob_i0_t2 = smooth_func(0, 5, 1, 10, lamb)[0]
        prob_i0_t3 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t4 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t5 = smooth_func(0, 5, 1, 10, lamb)[0]
        
        self.assertEquals(p.prob_tag_given_item(0, 0), prob_i0_t0)
        self.assertEquals(p.prob_tag_given_item(0, 1), prob_i0_t1)
        self.assertEquals(p.prob_tag_given_item(0, 2), prob_i0_t2)
        self.assertEquals(p.prob_tag_given_item(0, 3), prob_i0_t3)
        self.assertEquals(p.prob_tag_given_item(0, 4), prob_i0_t4)
        self.assertEquals(p.prob_tag_given_item(0, 5), prob_i0_t5)
    
        self.assertEquals(p.log_prob_tag_given_item(0, 0), np.log2(prob_i0_t0))
        self.assertEquals(p.log_prob_tag_given_item(0, 1), np.log2(prob_i0_t1))
        self.assertEquals(p.log_prob_tag_given_item(0, 2), np.log2(prob_i0_t2))
        self.assertEquals(p.log_prob_tag_given_item(0, 3), np.log2(prob_i0_t3))
        self.assertEquals(p.log_prob_tag_given_item(0, 4), np.log2(prob_i0_t4))
        self.assertEquals(p.log_prob_tag_given_item(0, 5), np.log2(prob_i0_t5))
    
        #Vect methods
        expected = np.array([prob_i0_t0, prob_i0_t1, prob_i0_t2, prob_i0_t3, prob_i0_t4, prob_i0_t5])
        estimated = p.vect_prob_tag_given_item(0, range(6))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2([prob_i0_t0, prob_i0_t1, prob_i0_t2, prob_i0_t3, prob_i0_t4, prob_i0_t5])
        estimated = p.vect_log_prob_tag_given_item(0, range(6))
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())
    
    def test_prob_user(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.jelinek_mercer
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
        
    def test_bayes(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = smooth.bayes
        lamb = 0.3
        p = SmoothEstimator(smooth_func, lamb, self.annots)
        
        prob_i0_t0, alpha = smooth_func(2, 5, 3, 10, lamb)
        prob_i0_t1 = smooth_func(1, 5, 3, 10, lamb)[0]
        prob_i0_t2 = alpha * 1 / p.n_annotations
        prob_i0_t3 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t4 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t5 = alpha * 1 / p.n_annotations
        
        self.assertAlmostEquals(p.prob_tag_given_item(0, 0), prob_i0_t0)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 1), prob_i0_t1)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 2), prob_i0_t2)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 3), prob_i0_t3)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 4), prob_i0_t4)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 5), prob_i0_t5)
        
if __name__ == "__main__":
    unittest.main()