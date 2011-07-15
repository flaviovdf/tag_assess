# -*- coding: utf8

from __future__ import division, print_function

from tagassess.stats import topk
import unittest

class TestAll(unittest.TestCase):

    def test_all(self):
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
        
        self.assertEquals(0, topk.kendall_tau_distance(list_page_rank, 
                                                       list_page_rank))
        self.assertEquals(0, topk.kendall_tau_distance(list_wkpaths, 
                                                       list_wkpaths))
        self.assertEquals(0, topk.kendall_tau_distance(list_markov_c, 
                                                       list_markov_c))
        
        self.assertEquals(0.45, topk.kendall_tau_distance(list_page_rank, 
                                                          list_page_rank[::-1]))
        self.assertEquals(0.45, topk.kendall_tau_distance(list_wkpaths, 
                                                          list_wkpaths[::-1]))
        self.assertEquals(0.45, topk.kendall_tau_distance(list_markov_c, 
                                                          list_markov_c[::-1]))
        
        self.assertAlmostEquals(0.06, topk.kendall_tau_distance(list_page_rank, 
                                                                list_wkpaths), 2)
        self.assertAlmostEquals(0.68, topk.kendall_tau_distance(list_page_rank, 
                                                                list_markov_c), 2)
        
        self.assertAlmostEquals(0.06, topk.kendall_tau_distance(list_wkpaths, 
                                                                list_page_rank), 2)
        self.assertAlmostEquals(0.71, topk.kendall_tau_distance(list_wkpaths, 
                                                                list_markov_c), 2)
        
        self.assertAlmostEquals(0.68, topk.kendall_tau_distance(list_markov_c, 
                                                                list_page_rank), 2)
        self.assertAlmostEquals(0.71, topk.kendall_tau_distance(list_markov_c, 
                                                                list_wkpaths), 2)
        
        self.assertEquals(1, topk.kendall_tau_distance(list_page_rank, 
                                                       range(20, 30)))
        self.assertEquals(1, topk.kendall_tau_distance(list_wkpaths, 
                                                       range(20, 30)))
        self.assertEquals(1, topk.kendall_tau_distance(list_markov_c, 
                                                       range(20, 30)))