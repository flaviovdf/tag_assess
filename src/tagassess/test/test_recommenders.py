# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
from __future__ import division, print_function

from collections import defaultdict
from tagassess.recommenders import ProbabilityReccomender
from tagassess.probability_estimates import ProbabilityEstimator

import numpy as np
import unittest

class TestProbabilityReccomender(unittest.TestCase):

    def test_all(self):
        estimator = FakeEstimator()
        
        reccomender = ProbabilityReccomender(estimator)
        self.assertEquals(np.log2(0.95) + np.log2(0.25), reccomender.relevance(0, 1))
        self.assertEquals(np.log2(0.05) + np.log2(0.75), reccomender.relevance(0, 0))

class FakeEstimator(ProbabilityEstimator):
    
    def __init__(self):
        super(FakeEstimator, self).__init__()
        
    def prob_tag(self, tag):
        pass
    
    def prob_tag_given_item(self, item, tag):
        pass
    
    def prob_user(self, user):
        pass
    
    def prob_user_given_item(self, item, user):
        x = defaultdict(lambda: defaultdict(float))
        
        x[0][0] = 0.05
        x[1][0] = 0.95
        
        return x[item][user]
        
    def prob_item(self, item):
        x = defaultdict(lambda: defaultdict(float))
        
        x[0] = 0.75
        x[1] = 0.25
        
        return x[item]

    def num_items(self):
        pass

    def num_tags(self):
        pass
    
    def num_users(self):
        pass
    
    def num_annotations(self):
        pass
        
if __name__ == "__main__":
    unittest.main()