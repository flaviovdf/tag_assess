# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111

from __future__ import print_function, division

from tagassess import mitagvalue

import math
import unittest

#Calculates the entropy iteratively.
def it_entropy(probs):
    entropy = 0.0
    for prob in probs:
        entropy -= prob * math.log(prob, 2)
    return entropy

class TestEntropyMI(unittest.TestCase):
    '''
    Tests entropy by comparing the return
    with an iterative calculation
    '''
    def test_entropy(self):
        probs = [0.1, 0.5, 0.01, 0.07, 0.02, 0.3]

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

        try:
            mitagvalue.entropy([])
            self.fail()
        except AssertionError:
            pass

    def test_norm_mi(self):
        x_probs = [0.04, 0.16] * 5
        xy_probs = [0.02, 0.18] * 5

        h_x = it_entropy(x_probs)
        h_y = it_entropy(xy_probs)

        mi = 1 - (h_x - h_y)/h_x
        self.assertEqual(mitagvalue.norm_mutual_information(x_probs, xy_probs), mi)

        x_probs = [1]
        self.assertEqual(mitagvalue.norm_mutual_information(x_probs, xy_probs), 0)

if __name__ == "__main__":
    unittest.main()