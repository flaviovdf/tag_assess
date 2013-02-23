# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=W0212

from __future__ import print_function, division

import numpy as np

from tagassess import data_parser
from tagassess.probability_estimates.smooth_estimator import SmoothEstimator
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
        return est, value_calculator.ValueCalculator(est)
    
    def __init_test(self, annot_file):
        parser = data_parser.Parser()
        with open(annot_file) as in_f:
            for annot in parser.iparse(in_f, 
                                       data_parser.delicious_flickr_parser):
                self.annots.append(annot)
    
    def test_valid_values_personalized(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        items = np.arange(5)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_personalized(0, items):
            self.assertTrue(val >= 0)

    def test_valid_values_per_user(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        items = np.arange(5)
        
        for val in vc.tag_value_per_user_search(0, items):
            self.assertTrue(val >= 0)

    def test_valid_values_item_search(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.build_value_calculator(self.annots, smooth_func, lambda_)
        
        items = np.arange(5)
        
        for val in vc.tag_value_item_search(items):
            self.assertTrue(val >= 0)