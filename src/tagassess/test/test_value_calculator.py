# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=W0212

from __future__ import print_function, division

from numpy import log2
from tagassess import value_calculator, smooth
from tagassess import data_parser
from tagassess import test
from tagassess.dao import annotations
from tagassess.probability_estimates import SmoothedItemsUsersAsTags  

import os
import tempfile
import unittest

class TestAll(unittest.TestCase):
    
    def setUp(self):
        self.h5_file = None

    def __init_test(self, fpath):
        self.h5_file = tempfile.mktemp('testw.h5')
        parser = data_parser.Parser()
        with open(fpath) as in_f:
            with annotations.AnnotWriter(self.h5_file) as writer:
                writer.create_table('deli')
                for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                    writer.write(annot)
                    
    def tearDown(self):
        if self.h5_file and os.path.exists(self.h5_file):
            os.remove(self.h5_file)
    
    def test_items_and_user_items(self):
        self.__init_test(test.SMALL_DEL_FILE)
        items, uitems0 = \
            value_calculator._items_and_user_items(self.h5_file, 'deli', 0)
        
        self.assertEquals(items, set(xrange(5)))
        self.assertEquals(uitems0, set(xrange(3)))
        
        uitems1 = \
            value_calculator._items_and_user_items(self.h5_file, 'deli', 1)[1]
        self.assertEquals(uitems1, set([0, 3, 4]))
            
        uitems2 = \
            value_calculator._items_and_user_items(self.h5_file, 'deli', 2)[1]
        self.assertEquals(uitems2, set([0, 2]))
        
    def test_tags_and_user_tags(self):
        self.__init_test(test.SMALL_DEL_FILE)
        tags, utags0 = \
            value_calculator._tags_and_user_tags(self.h5_file, 'deli', 0)
        
        self.assertEquals(tags, set(xrange(6)))
        self.assertEquals(utags0, set(xrange(3)))
        
        utags1 = \
            value_calculator._tags_and_user_tags(self.h5_file, 'deli', 1)[1]
        self.assertEquals(utags1, set([0, 1, 3]))
            
        utags2 = \
            value_calculator._tags_and_user_tags(self.h5_file, 'deli', 2)[1]
        self.assertEquals(utags2, set([4, 5]))
    
    def test_iitem_value(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.bayes
        lambda_ = 0.3
        
        estimator = SmoothedItemsUsersAsTags(self.h5_file, 'deli', 
                                             smooth_func, lambda_)
        estimator.open()
        
        for user in [0, 1, 2]:
            tag_vals = dict((v, k) for k, v, b in value_calculator.itag_value(self.h5_file, 'deli', user, smooth_func, lambda_, -1, False))
            
            for tag in [0, 1, 2, 3, 4, 5]:
                #Iterative calculation
                pt = estimator.prob_tag(tag)
                pu = estimator.prob_user(user)
                
                val = 0
                for item in [0, 1, 2, 3, 4]:
                    pi = estimator.prob_item(item)
                    pti = estimator.prob_tag_given_item(item, tag)
                    pui = estimator.prob_user_given_item(item, user)
                    
                    val += pui * pti * pi * log2(pti / pt)
                val /= pt * pu
                
                #Assert
                self.assertAlmostEquals(tag_vals[tag], val)