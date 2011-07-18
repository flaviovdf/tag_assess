# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
from __future__ import division, print_function
import pyximport; pyximport.install()

from tagassess import smooth

import unittest

class TestSmooth(unittest.TestCase):

    def test_jm(self):
        self.assertAlmostEquals(smooth.jelinek_mercer(10, 100, 20, 500, 0.6)[0], 0.064)
        self.assertAlmostEquals(smooth.jelinek_mercer(10, 100, 20, 500, 0.5)[0], 0.07)
        self.assertAlmostEquals(smooth.jelinek_mercer(1, 1, 1, 1, 1)[0], 1)

    def test_bayes(self):
        self.assertAlmostEquals(smooth.bayes(10, 100, 20, 500, 0.1)[0], .0999, 4)
        self.assertAlmostEquals(smooth.bayes(10, 100, 20, 500, 0.5)[0], .0997, 4)
        self.assertEquals(smooth.bayes(1, 1, 1, 1, 1)[0], 1)
        
        self.assertAlmostEquals(smooth.bayes(10, 100, 20, 500, 0.1)[1], 0.000999000999)
        self.assertAlmostEquals(smooth.bayes(10, 100, 20, 500, 0.5)[1], 0.00497512438)
        self.assertEquals(smooth.bayes(1, 1, 1, 1, 1)[1], 0.5)
    
if __name__ == "__main__":
    unittest.main()
