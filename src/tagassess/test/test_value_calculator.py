# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=W0212

from __future__ import print_function, division

import numpy as np

from tagassess import data_parser
from tagassess import test
from tagassess import value_calculator

from tagassess.stats.topk import kendall_tau_distance as dktau
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator

import unittest

class TestValueCaculator(unittest.TestCase):
    
    def setUp(self):
        self.annots = []
    
    def tearDown(self):
        self.annots = None
        
    def build_value_calculator(self, annots, smooth_func, lambda_):
        est = SmoothEstimator(smooth_func, lambda_, annots, 1)
        return est, value_calculator.ValueCalculator(est, annots)
    
    def __init_test(self, annot_file):
        parser = data_parser.Parser()
        with open(annot_file) as in_f:
            for annot in parser.iparse(in_f, 
                                       data_parser.delicious_flickr_parser):
                self.annots.append(annot)
    
    def test_rho(self):
        self.__init_test(test.SMALL_DEL_FILE)
    
        smooth_func = 'Bayes'
        lambda_ = 0.1
        _, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        rho = vc.calc_rho(0, np.array([2.0, 0, 10]), np.array([0, 1, 2]))
        self.assertEqual(rho, 1 - dktau([2, 0, 1], [2, 0], k=2, p=1))
    
    
    def test_rho_inverse(self):
        self.__init_test(test.SMALL_DEL_FILE)
    
        smooth_func = 'Bayes'
        lambda_ = 0.1
        _, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        rho = vc.calc_rho(0, np.array([10, 0, 2.0]), np.array([0, 1, 2]))
        self.assertEqual(rho, 1 - dktau([0, 2, 1], [0, 2], k=2, p=1))

    def test_rho_one_more(self):
        self.__init_test(test.SMALL_DEL_FILE)
    
        smooth_func = 'Bayes'
        lambda_ = 0.1
        _, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        rho = vc.calc_rho(0, np.array([0, 20, 10.0]), np.array([0, 1, 2]))
        self.assertEqual(rho, 1 - dktau([1, 2, 0], [2, 0], k=2, p=1))
    
    def test_rho_one_item(self):
        self.__init_test(test.SMALL_DEL_FILE)
    
        smooth_func = 'Bayes'
        lambda_ = 0.1
        _, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        rho = vc.calc_rho(0, np.array([0.0]), np.array([0]))
        self.assertEqual(rho, 1 - dktau([0], [0], k=1, p=1))
    
    def test_valid_values_personalized(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        items = np.arange(5)
        tags = np.arange(5)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        _, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_personalized(0, items, tags):
            self.assertTrue(val >= 0)

    def test_valid_values_item_search(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        _, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        items = np.arange(5)
        tags = np.arange(5)
        
        for val in vc.tag_value_item_search(items, tags):
            self.assertTrue(val >= 0)