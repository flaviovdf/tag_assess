# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=R0915
#pylint: disable-msg=W0212
from __future__ import division, print_function

from tagassess import data_parser
from tagassess import smooth
from tagassess import test
from tagassess.dao import annotations
from tagassess.probability_estimates import MLE, SmoothedItems, SmoothedItemsUsersAsTags

import os
import tempfile
import unittest

class TestMLE(unittest.TestCase):

    def setUp(self):
        self.annots = []
        self.h5_file = None

    def __init_test(self, fpath):
        self.h5_file = tempfile.mktemp('testw.h5')
        parser = data_parser.Parser()
        with open(fpath) as in_f:
            with annotations.AnnotWriter(self.h5_file) as writer:
                writer.create_table('deli')
                for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                    self.annots.append(annot)
                    
    def tearDown(self):
        if self.h5_file and os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        
    def test_all(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        p = MLE()
        p.open(self.annots)
        
        #Item frequencies
        self.assertEquals(p.item_col_freq[0], 5)
        self.assertEquals(p.item_col_freq[1], 1)
        self.assertEquals(p.item_col_freq[2], 2)
        self.assertEquals(p.item_col_freq[3], 1)
        self.assertEquals(p.item_col_freq[4], 1)
        
        #Item probabilities
        self.assertEquals(p.prob_item(0), 5 / 10)
        self.assertEquals(p.prob_item(1), 1 / 10)
        self.assertEquals(p.prob_item(2), 2 / 10)
        self.assertEquals(p.prob_item(3), 1 / 10)
        self.assertEquals(p.prob_item(4), 1 / 10)
        
        #Tag freqs
        self.assertEquals(p.tag_col_freq[0], 3)
        self.assertEquals(p.tag_col_freq[1], 3)
        self.assertEquals(p.tag_col_freq[2], 1)
        self.assertEquals(p.tag_col_freq[3], 1)
        self.assertEquals(p.tag_col_freq[4], 1)
        self.assertEquals(p.tag_col_freq[5], 1)
        
        #Tag Prob
        self.assertEquals(p.prob_tag(0), 3 / 10)
        self.assertEquals(p.prob_tag(1), 3 / 10)
        self.assertEquals(p.prob_tag(2), 1 / 10)
        self.assertEquals(p.prob_tag(3), 1 / 10)
        self.assertEquals(p.prob_tag(4), 1 / 10)
        self.assertEquals(p.prob_tag(5), 1 / 10)
        
        #Item tag frequencies
        self.assertEquals(p.item_tag_freq[0][0], 2)
        self.assertEquals(p.item_tag_freq[0][1], 1)
        self.assertEquals(p.item_tag_freq[0][2], 0)
        self.assertEquals(p.item_tag_freq[0][3], 1)
        self.assertEquals(p.item_tag_freq[0][4], 1)
        self.assertEquals(p.item_tag_freq[0][5], 0)
        
        #Prob P(t|i)
        self.assertEquals(p.prob_tag_given_item(0, 0), 2 / 5)
        self.assertEquals(p.prob_tag_given_item(0, 1), 1 / 5)
        self.assertEquals(p.prob_tag_given_item(0, 2), 0)
        self.assertEquals(p.prob_tag_given_item(0, 3), 1 / 5)
        self.assertEquals(p.prob_tag_given_item(0, 4), 1 / 5)
        self.assertEquals(p.prob_tag_given_item(0, 5), 0)
        
        #User frequencies
        self.assertEquals(p.user_col_freq[0], 4)
        self.assertEquals(p.user_col_freq[1], 4)
        self.assertEquals(p.user_col_freq[2], 2)

        #User probs
        self.assertEquals(p.prob_user(0), 4 / 10)
        self.assertEquals(p.prob_user(1), 4 / 10)
        self.assertEquals(p.prob_user(2), 2 / 10)

        #Item user frequencies
        self.assertEquals(p.item_user_freq[0][0], 2)
        self.assertEquals(p.item_user_freq[1][0], 1)
        self.assertEquals(p.item_user_freq[1][1], 0)
        self.assertEquals(p.item_user_freq[1][2], 0)
        
        #Item user probs
        self.assertEquals(p.prob_user_given_item(0, 0), 2 / 5)
        self.assertEquals(p.prob_user_given_item(1, 0), 1)
        self.assertEquals(p.prob_user_given_item(1, 1), 0)
        self.assertEquals(p.prob_user_given_item(1, 2), 0)
 
class TestSmoothedItems(unittest.TestCase):
    
    def setUp(self):
        self.annots = []
        self.h5_file = None

    def __init_test(self, fpath):
        self.h5_file = tempfile.mktemp('testw.h5')
        parser = data_parser.Parser()
        with open(fpath) as in_f:
            with annotations.AnnotWriter(self.h5_file) as writer:
                writer.create_table('deli')
                for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                    self.annots.append(annot)
    
    def tearDown(self):
        if self.h5_file and os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        
    def test_all_jm(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = smooth.jelinek_mercer
        lamb = 0.5
        p = SmoothedItems(smooth_func, lamb)
        p.open(self.annots)

        prob_i0_t0 = smooth_func(2, 5, 3, 10, lamb)[0]
        prob_i0_t1 = smooth_func(1, 5, 3, 10, lamb)[0]
        prob_i0_t2 = smooth_func(0, 5, 1, 10, lamb)[0]
        prob_i0_t3 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t4 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t5 = smooth_func(0, 5, 1, 10, lamb)[0]
        
        self.assertEquals(p.prob_tag_given_item(0, 0), prob_i0_t0)
        self.assertEquals(p.prob_tag_given_item(0, 1), prob_i0_t1)
        self.assertEquals(p.prob_tag_given_item(0, 2), prob_i0_t2)
        self.assertEquals(p.prob_tag_given_item(0, 3), prob_i0_t3)
        self.assertEquals(p.prob_tag_given_item(0, 4), prob_i0_t4)
        self.assertEquals(p.prob_tag_given_item(0, 5), prob_i0_t5)
        
        prob_t0 = sum(p.prob_tag_given_item(i, 0) * p.prob_item(i) for i in xrange(5))
        prob_t1 = sum(p.prob_tag_given_item(i, 1) * p.prob_item(i) for i in xrange(5))
        prob_t2 = sum(p.prob_tag_given_item(i, 2) * p.prob_item(i) for i in xrange(5))
        prob_t3 = sum(p.prob_tag_given_item(i, 3) * p.prob_item(i) for i in xrange(5))
        prob_t4 = sum(p.prob_tag_given_item(i, 4) * p.prob_item(i) for i in xrange(5))
        prob_t5 = sum(p.prob_tag_given_item(i, 5) * p.prob_item(i) for i in xrange(5))
        
        self.assertEquals(p.prob_tag(0), prob_t0)
        self.assertEquals(p.prob_tag(1), prob_t1)
        self.assertEquals(p.prob_tag(2), prob_t2)
        self.assertEquals(p.prob_tag(3), prob_t3)
        self.assertEquals(p.prob_tag(4), prob_t4)
        self.assertEquals(p.prob_tag(5), prob_t5)
        
    def test_all_bayes(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = smooth.bayes
        lamb = 0.3
        p = SmoothedItems(smooth_func, lamb)
        p.open(self.annots)
        
        prob_i0_t0, alpha = smooth_func(2, 5, 3, 10, lamb)
        prob_i0_t1 = smooth_func(1, 5, 3, 10, lamb)[0]
        prob_i0_t2 = alpha * p.tag_col_freq[2] / p.n_annotations
        prob_i0_t3 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t4 = smooth_func(1, 5, 1, 10, lamb)[0]
        prob_i0_t5 = alpha * p.tag_col_freq[5] / p.n_annotations
        
        self.assertAlmostEquals(p.prob_tag_given_item(0, 0), prob_i0_t0)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 1), prob_i0_t1)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 2), prob_i0_t2)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 3), prob_i0_t3)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 4), prob_i0_t4)
        self.assertAlmostEquals(p.prob_tag_given_item(0, 5), prob_i0_t5)
        
class TestSmoothedItemsUsersAsTags(unittest.TestCase):
    
    def setUp(self):
        self.annots = []
        self.h5_file = None

    def __init_test(self, fpath):
        self.h5_file = tempfile.mktemp('testw.h5')
        parser = data_parser.Parser()
        with open(fpath) as in_f:
            with annotations.AnnotWriter(self.h5_file) as writer:
                writer.create_table('deli')
                for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                    self.annots.append(annot)
                    
    def tearDown(self):
        if self.h5_file and os.path.exists(self.h5_file):
            os.remove(self.h5_file)
        
    def test_all(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = smooth.jelinek_mercer
        lamb = 0.5
        p = SmoothedItemsUsersAsTags(smooth_func, lamb)
        p.open(self.annots)
        
        prob = p.prob_user(0)
        expected_prob = p.prob_tag(0) * p.prob_tag(1) * p.prob_tag(2) 
        self.assertEquals(prob, expected_prob)

        prob = p.prob_user_given_item(0, 0)
        expected_prob = p.prob_tag_given_item(0, 0) * p.prob_tag_given_item(0, 1) * p.prob_tag_given_item(0, 2) 
        self.assertEquals(prob, expected_prob)
                
if __name__ == "__main__":
    unittest.main()