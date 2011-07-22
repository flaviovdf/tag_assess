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
from tagassess.probability_estimates import SmoothEstimator
from tagassess.recommenders import ProbabilityReccomender
from tagassess.test import PyCyUnit

class TestValueCaculator(PyCyUnit):
    
    def get_module_to_test(self):
        from tagassess import value_calculator
        return value_calculator
    
    def __init_test(self, annot_file):
        parser = data_parser.Parser()
        with open(annot_file) as in_f:
            for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                self.annots.append(annot)
    
    def setUp(self):
        super(TestValueCaculator, self).setUp()
        self.annots = []
    
    def tearDown(self):
        self.annots = None
        
    def test_itag_value_user(self):
        value_calculator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est = SmoothEstimator(smooth_func, lambda_, self.annots)
        recc = ProbabilityReccomender(est)
        
        vc = value_calculator.ValueCalculator(est, recc)
        
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
            tag_vals = vc.tag_value_ucontext(user)
            
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

    def test_itag_value_user_fiter_items(self):
        value_calculator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est = SmoothEstimator(smooth_func, lambda_, self.annots)
        recc = ProbabilityReccomender(est)
        
        vc = value_calculator.ValueCalculator(est, recc)
        
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
            tag_vals = vc.tag_value_ucontext(user, np.array([0, 1, 2]))
            
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

    def test_itag_value_global(self):
        value_calculator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'Bayes'
        lambda_ = 0.3
        est = SmoothEstimator(smooth_func, lambda_, self.annots)
        recc = ProbabilityReccomender(est)
        
        vc = value_calculator.ValueCalculator(est, recc)
        
        tag_vals = vc.tag_value_gcontext()
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
        value_calculator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est = SmoothEstimator(smooth_func, lambda_, self.annots)
        recc = ProbabilityReccomender(est)
        vc = value_calculator.ValueCalculator(est, recc)
        
        for item, val in vc.item_value(0).items():
            self.assertTrue(val < 0)
            
    def test_valid_values_user(self):
        value_calculator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est = SmoothEstimator(smooth_func, lambda_, self.annots)
        recc = ProbabilityReccomender(est)
        vc = value_calculator.ValueCalculator(est, recc)
        
        for tag, val in vc.tag_value_ucontext(0).items():
            self.assertTrue(val >= 0)

    def test_valid_values_global(self):
        value_calculator = self.mod_under_test
        
        self.__init_test(test.SMALL_DEL_FILE)
        smooth_func = 'Bayes'
        lambda_ = 0.1
        est = SmoothEstimator(smooth_func, lambda_, self.annots)
        recc = ProbabilityReccomender(est)
        vc = value_calculator.ValueCalculator(est, recc)
        
        for tag, val in vc.tag_value_gcontext().items():
            self.assertTrue(val >= 0)