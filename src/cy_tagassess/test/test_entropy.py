# -*- coding: utf8
from __future__ import division, print_function

import pyximport; pyximport.install()

from tagassess.test.test_entropy import TestEntropy

class CyTestEntropy(TestEntropy):
    
    def get_module_to_test(self):
        from cy_tagassess import entropy
        return entropy