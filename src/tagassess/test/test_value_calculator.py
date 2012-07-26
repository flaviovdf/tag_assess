# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=W0212

from __future__ import print_function, division

import numpy as np

from tagassess import data_parser
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator
from tagassess.probability_estimates.base import DecoratorEstimator
from tagassess import test
from tagassess import value_calculator

import unittest

class TestValueCaculator(unittest.TestCase):
    
    def setUp(self):
        self.annots = []
    
    def tearDown(self):
        self.annots = None
        
    def build_value_calculator(self, annots, smooth_func, lambda_):
        est = SmoothEstimator(smooth_func, lambda_, annots)
        return DecoratorEstimator(est), value_calculator.ValueCalculator(est)
    
    def __init_test(self, annot_file):
        parser = data_parser.Parser()
        with open(annot_file) as in_f:
            for annot in parser.iparse(in_f, 
                                       data_parser.delicious_flickr_parser):
                self.annots.append(annot)
    
    def test_itag_value_personalized(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        pus = []
        s_pus = 0.0
        for user in [0, 1, 2]:
            pu = est.prob_user(user)
            pus.append(pu)
            s_pus += pu
        
        #Iterative calculation
        for i, pu in enumerate(pus):
            pus[i] = pu / s_pus
        
        for user in [0, 1, 2]:
            pu = pus[user]
            tag_vals = vc.tag_value_personalized(user)
            
            for tag in [0, 1, 2, 3, 4, 5]:
                pt = est.prob_tag(tag)
                
                pitus = []
                pius = []
                for item in [0, 1, 2, 3, 4]:
                    pi = est.prob_item(item)
                    pti = est.prob_tag_given_item(item, tag)
                    pui = est.prob_user_given_item(item, user)
                    
                    piu = pui * pi / pu
                    pitu = pti * pui * pi / (pu * pt)
                    
                    pitus.append(pitu)
                    pius.append(piu)
                
                val = 0
                for item in [0, 1, 2, 3, 4]:
                    n_pitu = pitus[item] / sum(pitus)
                    n_piu = pius[item] / sum(pius)
                    
                    val += n_pitu * np.log2(n_pitu / n_piu)
                
                #Assert
                self.assertAlmostEquals(tag_vals[tag], val)

    def test_itag_value_personalized_filter(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        pus = []
        s_pus = 0.0
        for user in [0, 1, 2]:
            pu = est.prob_user(user)
            pus.append(pu)
            s_pus += pu
        
        #Iterative calculation
        for i, pu in enumerate(pus):
            pus[i] = pu / s_pus
        
        for user in [0, 1, 2]:
            pu = pus[user]
            tag_vals = vc.tag_value_personalized(user, np.array([0, 1, 2]))
            
            for tag in [0, 1, 2, 3, 4, 5]:
                pt = est.prob_tag(tag)
                
                pitus = []
                pius = []
                for item in [0, 1, 2]:
                    pi = est.prob_item(item)
                    pti = est.prob_tag_given_item(item, tag)
                    pui = est.prob_user_given_item(item, user)
                    
                    piu = pui * pi / pu
                    pitu = pti * pui * pi / (pu * pt)
                    
                    pitus.append(pitu)
                    pius.append(piu)
                
                val = 0
                for item in [0, 1, 2]:
                    n_pitu = pitus[item] / sum(pitus)
                    n_piu = pius[item] / sum(pius)
                    
                    val += n_pitu * np.log2(n_pitu / n_piu)
                
                #Assert
                self.assertAlmostEquals(tag_vals[tag], val)

    def test_itag_value_item_search(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        tag_vals = vc.tag_value_item_search()
        for tag in [0, 1, 2, 3, 4, 5]:
            #Iterative calculation
            pt = est.prob_tag(tag)
            
            val = 0
            for item in [0, 1, 2, 3, 4]:
                pi = est.prob_item(item)
                pti = est.prob_tag_given_item(item, tag)
                
                val += pti * pi * np.log2(pti / pt)
            val /= pt
            
            #Assert
            self.assertAlmostEquals(tag_vals[tag], val)

    def test_valid_values_personalized(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_personalized(0):
            self.assertTrue(val >= 0)

    def test_valid_values_per_user(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_per_user_search(0):
            self.assertTrue(val >= 0)

    def test_valid_values_item_search(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_item_search():
            self.assertTrue(val >= 0)