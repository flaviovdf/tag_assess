# -*- coding: utf8

from __future__ import division, print_function

from tagassess.stats import topk
import unittest

class TestAll(unittest.TestCase):

    def test_different_penalty(self):
        '''Tests different penalty values'''
        
        l1 = [1, 2, 3, 4]
        l2 = [2, 5, 4, 3]
        
        dktau_p1 = topk.kendall_tau_distance(l1, l2, p=1)
        dktau_p0 = topk.kendall_tau_distance(l1, l2, p=0)
        dktau_p05 = topk.kendall_tau_distance(l1, l2, p=0.5)
        
        self.assertTrue(dktau_p1 != dktau_p0)
        self.assertTrue(dktau_p1 != dktau_p05)
        self.assertTrue(dktau_p0 != dktau_p05)

    def test_different_k(self):
        '''Tests different k values'''
        
        l1 = [1, 2, 3, 4]
        l2 = [2, 5, 4, 3]

        dktau_k2 = topk.kendall_tau_distance(l1, l2, k=2)
        dktau_k3 = topk.kendall_tau_distance(l1, l2, k=3)
        dktau_kneg1 = topk.kendall_tau_distance(l1, l2, k=-1)
        
        self.assertTrue(dktau_k2 != dktau_k3)
        self.assertTrue(dktau_k2 != dktau_kneg1)
        self.assertTrue(dktau_k3 != dktau_kneg1)

    def test_all_paper_values(self):
        '''
        Tests based on the datasets by the paper:
        [1] Algorithms for Estimating Relative Importance in Networks.
           Scott White, Padhraic Smyth
           KDD 2003
        
        Values will not match the paper exactly because we use a different
        normalization.
        '''
        Khemais = 1
        Beghal = 2
        Moussaoui = 3
        Maaroufi = 4
        Qatada = 5
        Daoudi = 6
        Courtaillier = 7 
        Bensakhria = 8
        Walid = 9
        Khammoun = 10
        Atta = 11
        Al_Shehhi = 12
        al_Shibh = 13
        Jarrah = 14
        Hanjour = 15
        Al_Omari = 16
        Bahaji = 17
            
        list_page_rank = [
            Khemais,
            Beghal,
            Moussaoui,
            Maaroufi,
            Qatada,
            Daoudi,
            Courtaillier, 
            Bensakhria,
            Walid,
            Khammoun 
        ]
        
        list_wkpaths = [
            Beghal,
            Khemais,
            Moussaoui,
            Maaroufi,
            Bensakhria, 
            Daoudi,
            Qatada,
            Walid,
            Courtaillier, 
            Khammoun 
        ]
        
        list_markov_c = [
            Atta,
            Al_Shehhi,
            al_Shibh,
            Moussaoui, 
            Jarrah,
            Hanjour,
            Al_Omari,
            Khemais,
            Qatada,
            Bahaji
        ]
        
        self.assertEquals(0, 
            topk.kendall_tau_distance(list_page_rank, list_page_rank))
        
        self.assertEquals(0, 
            topk.kendall_tau_distance(list_wkpaths, list_wkpaths))
        
        self.assertEquals(0, 
            topk.kendall_tau_distance(list_markov_c, list_markov_c))
        
        self.assertEquals(0.45, 
            topk.kendall_tau_distance(list_page_rank, list_page_rank[::-1]))
        
        self.assertEquals(0.45, 
            topk.kendall_tau_distance(list_wkpaths, list_wkpaths[::-1]))
        
        self.assertEquals(0.45, 
            topk.kendall_tau_distance(list_markov_c, list_markov_c[::-1]))
        
        self.assertAlmostEquals(0.06, 
            topk.kendall_tau_distance(list_page_rank, list_wkpaths), 2)
        
        self.assertAlmostEquals(0.68, 
            topk.kendall_tau_distance(list_page_rank, list_markov_c), 2)
        
        self.assertAlmostEquals(0.06, 
                topk.kendall_tau_distance(list_wkpaths, list_page_rank), 2)
        
        self.assertAlmostEquals(0.71, 
                topk.kendall_tau_distance(list_wkpaths, list_markov_c), 2)
        
        self.assertAlmostEquals(0.68, 
                topk.kendall_tau_distance(list_markov_c, list_page_rank), 2)
        
        self.assertAlmostEquals(0.71, 
                topk.kendall_tau_distance(list_markov_c, list_wkpaths), 2)
        
        self.assertEquals(1, 
                topk.kendall_tau_distance(list_page_rank, range(20, 30)))
        
        self.assertEquals(1, 
                topk.kendall_tau_distance(list_wkpaths, range(20, 30)))
        
        self.assertEquals(1, 
                topk.kendall_tau_distance(list_markov_c, range(20, 30)))        