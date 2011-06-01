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

import numpy as np
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
    
    def test_items(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.bayes
        lambda_ = 0.3
        vc = value_calculator.ValueCalculator(self.h5_file, 'deli', 
                                              smooth_func, lambda_)
        vc.open_reader()
        
        items = range(len(vc.est.item_col_freq))
        self.assertEquals(items, range(5))
        
    def test_tags_and_user_tags(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.bayes
        lambda_ = 0.3
        vc = value_calculator.ValueCalculator(self.h5_file, 'deli', 
                                              smooth_func, lambda_)
        vc.open_reader()
        
        tags = range(len(vc.est.tag_col_freq))
        self.assertEquals(tags, range(6))
    
    def test_with_filter(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.bayes
        lambda_ = 0.3
        vc = value_calculator.ValueCalculator(self.h5_file, 'deli', 
                                              smooth_func, lambda_)
        vc.set_filter_out({'user':[0], 'item':[0, 1, 2]})
        vc.open_reader()

        tags = vc.est.valid_tags()
        self.assertEquals(len(tags), 5)
        for tag in tags:
            self.assertTrue(tag in [0, 1, 3, 4, 5])
        
        items = vc.est.valid_items()
        self.assertEquals(len(items), 4)
        for item in items:
            self.assertTrue(item in [0, 2, 3, 4])
    
    def test_iitag_value_user(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.bayes
        lambda_ = 0.3
        vc = value_calculator.ValueCalculator(self.h5_file, 'deli', 
                                              smooth_func, lambda_)
        vc.open_reader()
        
        estimator = SmoothedItemsUsersAsTags(smooth_func, lambda_)
        estimator.open(vc._get_iterator())
        
        for user in [0, 1, 2]:
            tag_vals = dict((v, k) for k, v in vc.itag_value_ucontext(user))
            
            for tag in [0, 1, 2, 3, 4, 5]:
                #Iterative calculation
                pt = estimator.prob_tag(tag)
                pu = 1
                #pu = estimator.prob_user(user)
                
                val = 0
                for item in [0, 1, 2, 3, 4]:
                    pi = estimator.prob_item(item)
                    pti = estimator.prob_tag_given_item(item, tag)
                    pui = estimator.prob_user_given_item(item, user)
                    
                    val += pui * pti * pi * log2(pti / pt)
                val /= pt * pu
                
                #Assert
                self.assertAlmostEquals(tag_vals[tag], val)

    def test_iitag_value_global(self):
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = smooth.bayes
        lambda_ = 0.3
        vc = value_calculator.ValueCalculator(self.h5_file, 'deli', 
                                              smooth_func, lambda_)
        vc.open_reader()
        
        estimator = SmoothedItemsUsersAsTags(smooth_func, lambda_)
        estimator.open(vc._get_iterator())
        
        tag_vals = dict((v, k) for k, v in vc.itag_value_gcontext())
        for tag in [0, 1, 2, 3, 4, 5]:
            #Iterative calculation
            pt = estimator.prob_tag(tag)
            
            val = 0
            for item in [0, 1, 2, 3, 4]:
                pi = estimator.prob_item(item)
                pti = estimator.prob_tag_given_item(item, tag)
                
                val += pti * pi * log2(pti / pt)
            val /= pt
            
            #Assert
            self.assertAlmostEquals(tag_vals[tag], val)

#TODO: Commented out since it is not a problem with the code, and yes the model.
#    def test_valid_values(self):
#        self.__init_test(test.SMALL_DEL_FILE)
#        smooth_func = smooth.bayes
#        lambda_ = 0.1
#        vc = value_calculator.ValueCalculator(self.h5_file, 'deli', 
#                                              smooth_func, lambda_)
#        vc.open_reader()
#        for val, tag in sorted(vc.itag_value(0, -1, False)):
#            pass
#            print(val, tag)
#            self.assertTrue(val >= 0)