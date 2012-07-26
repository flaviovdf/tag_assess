# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
from __future__ import division, print_function

from tagassess.probability_estimates import smooth
from unittest import TestCase

import math

class TestSmooth(TestCase):

    def test_jm(self):
        self.assertTrue(math.isnan(smooth.jelinek_mercer(0, 0, 0, 0, 0)))
        
        self.assertEquals(smooth.jelinek_mercer(10, 100, 20, 500, 0.6), 0.064)
        self.assertEquals(smooth.jelinek_mercer(10, 100, 20, 500, 0.5), 0.07)
        self.assertEquals(smooth.jelinek_mercer(1, 1, 1, 1, 1), 1)

    def test_bayes(self):
        self.assertTrue(math.isnan(smooth.bayes(0, 0, 0, 0, 0)))
        
        self.assertAlmostEquals(smooth.bayes(10, 100, 20, 500, 0.1), .0999, 4)
        self.assertAlmostEquals(smooth.bayes(10, 100, 20, 500, 0.5), .0997, 4)
        self.assertEquals(smooth.bayes(1, 1, 1, 1, 1), 1)
        
        self.assertAlmostEquals(smooth.bayes(0, 100, 20, 500, 0.1), 
                                0.000999000999 * (20/500))