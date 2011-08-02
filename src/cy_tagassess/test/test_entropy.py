# -*- coding: utf8
from __future__ import division, print_function

from tagassess.test.test_entropy import TestEntropy

class CyTestEntropy(TestEntropy):
    
    def get_module_to_eval(self):
        from cy_tagassess import entropy
        return entropy