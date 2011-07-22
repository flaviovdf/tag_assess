# -*- coding: utf8
from __future__ import division, print_function

import pyximport; pyximport.install()

from tagassess.probability_estimates.test.test_smooth_estimator import TestSmoothEstimator

class CyTestEntropy(TestSmoothEstimator):
    
    def get_module_to_test(self):
        from cy_tagassess.probability_estimates import SmoothEstimator
        return SmoothEstimator