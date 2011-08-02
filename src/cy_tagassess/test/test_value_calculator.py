# -*- coding: utf8
from __future__ import division, print_function

from tagassess.test.test_value_calculator import TestValueCaculator

from tagassess.recommenders import ProbabilityReccomender
from cy_tagassess.probability_estimates import SmoothEstimator
from cy_tagassess import value_calculator

class CyTestValueCalculato(TestValueCaculator):
    
    def get_module_to_eval(self, *args, **kwargs):
        annots = args[0]
        smooth_func = args[1]
        lambda_ = args[2]
        
        est = SmoothEstimator(smooth_func, lambda_, annots)
        recc = ProbabilityReccomender(est)
        return est, value_calculator.ValueCalculator(est, recc)