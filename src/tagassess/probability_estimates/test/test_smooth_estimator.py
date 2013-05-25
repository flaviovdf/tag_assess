# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=R0915
#pylint: disable-msg=W0212

from __future__ import division, print_function

from tagassess.probability_estimates.smooth_estimator import SmoothEstimator
from tagassess.probability_estimates.smooth import bayes, jelinek_mercer

from tagassess import data_parser
from tagassess import test

from numpy.testing import assert_array_almost_equal

import numpy as np
import unittest

class TestSmoothEstimator(unittest.TestCase):

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
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)
        
        #Item probabilities
        self.assertAlmostEquals(p.prob_item(0), 5 / 10)
        self.assertAlmostEquals(p.prob_item(1), 1 / 10)
        self.assertAlmostEquals(p.prob_item(2), 2 / 10)
        self.assertAlmostEquals(p.prob_item(3), 1 / 10)
        self.assertAlmostEquals(p.prob_item(4), 1 / 10)
        
    def test_tag_given_item(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)
            
        #Tag given item
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

    def test_prob_user_given_item(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots)
        
        prob = p.prob_user_given_item(0, 0)
        self.assertTrue(prob > 0)
    
    def test_prob_user_given_item_profsize(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'JM'
        lamb = 0.5
        p = SmoothEstimator(smooth_func, lamb, self.annots, 
                user_profile_fract_size = 0)
        
        prob = p.prob_user_given_item(0, 0)
        self.assertEquals(prob, 0.0)
        
    def test_bayes(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lamb = 0.3
        p = SmoothEstimator(smooth_func, lamb, self.annots)
        
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

    def test_prob_items_given_user_and_tag(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        lambda_ = 0.3
        smooth_func = 'Bayes'
        p = SmoothEstimator(smooth_func, lambda_, self.annots)
        
        for user in [0, 1, 2]:
            for tag in [0, 1, 2, 3, 4, 5]:
                pitus = []
                pius = []
                for item in [0, 1, 2, 3, 4]:
                    pi = p.prob_item(item)
                    pti = p.prob_tag_given_item(item, tag)
                    pui = p.prob_user_given_item(item, user)
                    
                    piu = pui * pi
                    pitu = pti * pui * pi
                    
                    pitus.append(pitu)
                    pius.append(piu)
                
                sum_pitus = sum(pitus)
                sum_pius = sum(pius)
                for item in [0, 1, 2, 3, 4]:
                    pitus[item] = pitus[item] / sum_pitus
                    pius[item] = pius[item] / sum_pius
                    
                #Assert
                gamma_items = np.array([0, 1, 2, 3, 4])
                assert_array_almost_equal(pius, 
                        p.prob_items_given_user(user, gamma_items))
                assert_array_almost_equal(pitus, 
                        p.prob_items_given_user_tag(user, tag, gamma_items))
                
                self.assertAlmostEqual(1, sum(p.prob_items_given_user(user, 
                                                            gamma_items)))

                self.assertAlmostEqual(1, 
                        sum(p.prob_items_given_user_tag(user, tag, 
                                                            gamma_items)))             

    def test_prob_item_given_tag(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        lambda_ = 0.3
        smooth_func = 'Bayes'
        p = SmoothEstimator(smooth_func, lambda_, self.annots)
        
        for tag in [0, 1, 2, 3, 4, 5]:
            pis = []
            pits = []
            
            for item in [0, 1, 2, 3, 4]:
                pi = p.prob_item(item)
                pti = p.prob_tag_given_item(item, tag)
                
                pis.append(pi)
                pits.append(pti * pi)
                
            #Assert
            pis = np.array(pis)
            pis /= pis.sum()
            
            pits = np.array(pits)
            pits /= pits.sum()
            
            gamma_items = np.array([0, 1, 2, 3, 4])
            assert_array_almost_equal(pis, p.prob_items(gamma_items))
            assert_array_almost_equal(pits, p.prob_items_given_tag(tag, 
                                                                gamma_items))

            self.assertAlmostEqual(1, sum(p.prob_items(gamma_items)))
            self.assertAlmostEqual(1, 
                    sum(p.prob_items_given_tag(tag, gamma_items)))             


if __name__ == "__main__":
    unittest.main()
