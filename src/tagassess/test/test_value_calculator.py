# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=W0212

from __future__ import print_function, division

import numpy as np
from numpy import log2

from tagassess import data_parser
from tagassess import test
from tagassess import value_calculator
from tagassess.probability_estimates import SmoothEstimator
from tagassess.recommenders import ProbabilityReccomender
from tagassess.test import PyCyUnit

class TestValueCaculator(PyCyUnit):
    
    def setUp(self):
        self.annots = []
    
    def tearDown(self):
        self.annots = None
        
    def get_module_to_eval(self, *args, **kwargs):
        annots = args[0]
        smooth_func = args[1]
        lambda_ = args[2]
        
        est = SmoothEstimator(smooth_func, lambda_, annots)
        recc = ProbabilityReccomender(est)
        return est, value_calculator.ValueCalculator(est, recc)
    
    def __init_test(self, annot_file):
        parser = data_parser.Parser()
        with open(annot_file) as in_f:
            for annot in parser.iparse(in_f, 
                                       data_parser.delicious_flickr_parser):
                self.annots.append(annot)
    
    def test_itag_value_personalized(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est, vc = self.get_module_to_eval(self.annots, smooth_func, lambda_)
        
        pus = []
        s_pus = 0.0
        for user in [0, 1, 2]:
            pu = est.prob_user(user)
            pus.append(pu)
            s_pus += pu
        
        #Iterative calculation
        for i, pu in enumerate(pus):
            pus[i] = pu / s_pus
        
        for user in [0, 1, 2]:
            pu = pus[user]
            tag_vals = vc.tag_value_personalized(user)
            
            for tag in [0, 1, 2, 3, 4, 5]:
                pt = est.prob_tag(tag)
                
                pitus = []
                pius = []
                for item in [0, 1, 2, 3, 4]:
                    pi = est.prob_item(item)
                    pti = est.prob_tag_given_item(item, tag)
                    pui = est.prob_user_given_item(item, user)
                    
                    piu = pui * pi / pu
                    pitu = pti * pui * pi / (pu * pt)
                    
                    pitus.append(pitu)
                    pius.append(piu)
                
                val = 0
                for item in [0, 1, 2, 3, 4]:
                    n_pitu = pitus[item] / sum(pitus)
                    n_piu = pius[item] / sum(pius)
                    
                    val += n_pitu * log2(n_pitu / n_piu)
                
                #Assert
                self.assertAlmostEquals(tag_vals[tag], val)

    def test_itag_value_personalized_filter(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est, vc = self.get_module_to_eval(self.annots, smooth_func, lambda_)
        
        pus = []
        s_pus = 0.0
        for user in [0, 1, 2]:
            pu = est.prob_user(user)
            pus.append(pu)
            s_pus += pu
        
        #Iterative calculation
        for i, pu in enumerate(pus):
            pus[i] = pu / s_pus
        
        for user in [0, 1, 2]:
            pu = pus[user]
            tag_vals = vc.tag_value_personalized(user, np.array([0, 1, 2]))
            
            for tag in [0, 1, 2, 3, 4, 5]:
                pt = est.prob_tag(tag)
                
                pitus = []
                pius = []
                for item in [0, 1, 2]:
                    pi = est.prob_item(item)
                    pti = est.prob_tag_given_item(item, tag)
                    pui = est.prob_user_given_item(item, user)
                    
                    piu = pui * pi / pu
                    pitu = pti * pui * pi / (pu * pt)
                    
                    pitus.append(pitu)
                    pius.append(piu)
                
                val = 0
                for item in [0, 1, 2]:
                    n_pitu = pitus[item] / sum(pitus)
                    n_piu = pius[item] / sum(pius)
                    
                    val += n_pitu * log2(n_pitu / n_piu)
                
                #Assert
                self.assertAlmostEquals(tag_vals[tag], val)

    def test_itag_value_item_search(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est, vc = self.get_module_to_eval(self.annots, smooth_func, lambda_)
        
        tag_vals = vc.tag_value_item_search()
        for tag in [0, 1, 2, 3, 4, 5]:
            #Iterative calculation
            pt = est.prob_tag(tag)
            
            val = 0
            for item in [0, 1, 2, 3, 4]:
                pi = est.prob_item(item)
                pti = est.prob_tag_given_item(item, tag)
                
                val += pti * pi * log2(pti / pt)
            val /= pt
            
            #Assert
            self.assertAlmostEquals(tag_vals[tag], val)

    def test_valid_values_items(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.get_module_to_eval(self.annots, smooth_func, lambda_)
        
        for val in vc.item_value(0):
            self.assertTrue(val < 0)
            
    def test_valid_values_personalized(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.get_module_to_eval(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_personalized(0):
            self.assertTrue(val >= 0)

    def test_valid_values_per_user(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.get_module_to_eval(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_per_user_search(0):
            self.assertTrue(val >= 0)

    def test_valid_values_item_search(self):
        self.__init_test(test.SMALL_DEL_FILE)
        
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est, vc = self.get_module_to_eval(self.annots, smooth_func, lambda_)
        
        for val in vc.tag_value_item_search():
            self.assertTrue(val >= 0)