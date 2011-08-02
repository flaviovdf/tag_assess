# -*- coding: utf8
from __future__ import division, print_function

from tagassess.probability_estimates.test.test_smooth_estimator import TestSmoothEstimator

class CyTestSmoothEstimator(TestSmoothEstimator):
    
    def get_module_to_eval(self):
        from cy_tagassess.probability_estimates import SmoothEstimator
        return SmoothEstimator