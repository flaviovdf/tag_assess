# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=W0404

from __future__ import print_function, division

from tagassess import entropy

import math
import unittest

#Calculates the entropy iteratively.
def it_entropy(probs):
    ent = 0.0
    for prob in probs:
        if prob == 0:
            continue
        ent -= prob * math.log(prob, 2)
    return ent

class TestEntropy(unittest.TestCase):
    '''
    Tests entropy by comparing the return
    with an iterative calculation
    '''
    def test_entropy(self):
        probs = [0.1, 0.5, 0.01, 0.07, 0.02, 0.3, 0, 0, 0]

        self.assertEquals(entropy.entropy(probs), it_entropy(probs))

        try:
            entropy.entropy([-1])
            self.fail()
        except AssertionError:
            pass

        try:
            entropy.entropy([0.1, 0.8])
            self.fail()
        except AssertionError:
            pass

        try:
            entropy.entropy([2, -1])
            self.fail()
        except AssertionError:
            pass

        try:
            entropy.entropy([])
            self.fail()
        except AssertionError:
            pass

    def test_norm_mi(self):
        x_probs = [0.04, 0.16] * 5
        xy_probs = [0.02, 0.18] * 5

        h_x = it_entropy(x_probs)
        h_y = it_entropy(xy_probs)

        mutual_inf = 1 - (h_x - h_y)/h_x
        self.assertEqual(entropy.norm_mutual_information(x_probs, xy_probs), mutual_inf)

        x_probs = [1]
        self.assertEqual(entropy.norm_mutual_information(x_probs, xy_probs), 0)

    def test_mi(self):
        x_probs = [0.04, 0.16] * 5
        xy_probs = [0.02, 0.18] * 5

        h_x = it_entropy(x_probs)
        h_y = it_entropy(xy_probs)

        mutual_inf = h_x - h_y
        self.assertAlmostEqual(entropy.mutual_information(x_probs, xy_probs), mutual_inf)

    def test_kl(self):
        x_probs = [0.04, 0.16] * 5
        xy_probs = [0.02, 0.18] * 5
        
        dkl = 0
        for i in range(len(x_probs)):
            div = x_probs[i] / xy_probs[i]
            dkl += x_probs[i] * math.log(div, 2)
            
        self.assertAlmostEqual(entropy.kullback_leiber_divergence(x_probs, xy_probs), dkl)

    def test_kl2(self):
        x_probs = [0.04, 0.16] * 5 + [0]
        xy_probs = [0.02, 0.18] * 5 + [0]
        
        dkl = 0
        for i in range(len(x_probs) - 1):
            div = x_probs[i] / xy_probs[i]
            dkl += x_probs[i] * math.log(div, 2)
            
        self.assertAlmostEqual(entropy.kullback_leiber_divergence(x_probs, xy_probs), dkl)

    def test_kl3(self):
        x_probs = [0.25, 0.20, 0, 0.55]
        xy_probs = [0.20, 0, 0.25, 0.55]
        
        self.assertAlmostEqual(entropy.kullback_leiber_divergence(x_probs, xy_probs), float('inf'))

if __name__ == "__main__":
    unittest.main()