# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111

from __future__ import print_function, division

from tagassess import mitagvalue

import math
import unittest

class TestEntropyMI(unittest.TestCase):

    def test_entropy(self):
        '''Tests entropy by comparing the return
        with an iterative calculation'''
        probs = [0.1, 0.5, 0.01, 0.07, 0.02, 0.3]

        #Doing a iterative Calculation of entropy.
        def it_entropy(prob):
            entropy = 0.0
            for prob in probs:
                entropy -= prob * math.log(prob, 2)
            return entropy

        self.assertEquals(mitagvalue.entropy(probs), it_entropy(probs))

        try:
            mitagvalue.entropy([-1])
            self.fail()
        except AssertionError:
            pass

        try:
            mitagvalue.entropy([0.1, 0.8])
            self.fail()
        except AssertionError:
            pass

        try:
            mitagvalue.entropy([2, -1])
            self.fail()
        except AssertionError:
            pass

    def test_norm_mi(self):
        x_probs = [0.04, 0.16] * 5
        xy_probs = [0.02, 0.18] * 5

        h_x = mitagvalue.entropy(x_probs)
        h_y = mitagvalue.entropy(xy_probs)

        self.assertEqual(mitagvalue.norm_mutual_information(x_probs, xy_probs), 1 - (h_x - h_y)/h_x)

        x_probs = [1]
        self.assertEqual(mitagvalue.norm_mutual_information(x_probs, xy_probs), 0)

if __name__ == "__main__":
    unittest.main()