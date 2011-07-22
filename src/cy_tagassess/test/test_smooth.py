# -*- coding: utf8
from __future__ import division, print_function

from tagassess.test.test_smooth import TestSmooth

class CyTestSmooth(TestSmooth):
    
    def get_module_to_test(self):
        from cy_tagassess import smooth
        return smooth