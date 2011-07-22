# -*- coding: utf8
from __future__ import division, print_function

from tagassess.test.test_value_calculator import TestValueCaculator

class CyTestEntropy(TestValueCaculator):
    
    def get_module_to_test(self):
        from cy_tagassess import value_calculator
        return value_calculator