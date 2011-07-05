# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=R0915
#pylint: disable-msg=W0212
from __future__ import division, print_function

from tagassess import data_parser
from tagassess import graph
from tagassess import test
from tagassess.probability_estimates import RWEstimator

import networkx as nx
import numpy as np
import unittest

class TestAll(unittest.TestCase):

    def setUp(self):
        self.annots = []

    def __init_test(self, fpath):
        parser = data_parser.Parser()
        with open(fpath) as in_f:
            for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                self.annots.append(annot)
                    
    def tearDown(self):
        self.annots = None
        
    def test_prob_item(self):
        self.__init_test(test.SMALL_DEL_FILE)
        p = RWEstimator(self.annots)
        
        ntags, nitems, edges = graph.iedge_from_annotations(self.annots)
        nx_graph = graph.create_nxgraph(edges)
        
        pagerank_dict = nx.pagerank(nx_graph)
        pageranks = [pagerank_dict[node_id] for node_id in sorted(pagerank_dict)]
        
        item_pager = np.array(pageranks[ntags:])
        item_pager /= item_pager.sum()
        
        #Item probabilities
        self.assertAlmostEquals(p.prob_item(0), item_pager[0], 4)
        self.assertAlmostEquals(p.prob_item(1), item_pager[1], 4)
        self.assertAlmostEquals(p.prob_item(2), item_pager[2], 4)
        self.assertAlmostEquals(p.prob_item(3), item_pager[3], 4)
        self.assertAlmostEquals(p.prob_item(4), item_pager[4], 4)
        
        self.assertAlmostEquals(p.log_prob_item(0), np.log2(item_pager[0]), 4)
        self.assertAlmostEquals(p.log_prob_item(1), np.log2(item_pager[1]), 4)
        self.assertAlmostEquals(p.log_prob_item(2), np.log2(item_pager[2]), 4)
        self.assertAlmostEquals(p.log_prob_item(3), np.log2(item_pager[3]), 4)
        self.assertAlmostEquals(p.log_prob_item(4), np.log2(item_pager[4]), 4)
        
        #Vect methods
        estimated = p.vect_prob_item(range(5)).round(4)
        expected = item_pager.round(4)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2(item_pager).round(4)
        estimated = p.vect_log_prob_item(range(5)).round(4)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())
    
    def test_prob_tag(self):
        self.__init_test(test.SMALL_DEL_FILE)
        p = RWEstimator(self.annots)
        
        ntags, nitems, edges = graph.iedge_from_annotations(self.annots)
        nx_graph = graph.create_nxgraph(edges)
        
        pagerank_dict = nx.pagerank(nx_graph)
        pageranks = [pagerank_dict[node_id] for node_id in sorted(pagerank_dict)]
        
        tag_pager = np.array(pageranks[:ntags])
        tag_pager /= tag_pager.sum()
        
        #Item probabilities
        self.assertAlmostEquals(p.prob_tag(0), tag_pager[0], 4)
        self.assertAlmostEquals(p.prob_tag(1), tag_pager[1], 4)
        self.assertAlmostEquals(p.prob_tag(2), tag_pager[2], 4)
        self.assertAlmostEquals(p.prob_tag(3), tag_pager[3], 4)
        self.assertAlmostEquals(p.prob_tag(4), tag_pager[4], 4)
        self.assertAlmostEquals(p.prob_tag(5), tag_pager[5], 4)
        
        self.assertAlmostEquals(p.log_prob_tag(0), np.log2(tag_pager[0]), 4)
        self.assertAlmostEquals(p.log_prob_tag(1), np.log2(tag_pager[1]), 4)
        self.assertAlmostEquals(p.log_prob_tag(2), np.log2(tag_pager[2]), 4)
        self.assertAlmostEquals(p.log_prob_tag(3), np.log2(tag_pager[3]), 4)
        self.assertAlmostEquals(p.log_prob_tag(4), np.log2(tag_pager[4]), 4)
        self.assertAlmostEquals(p.log_prob_tag(5), np.log2(tag_pager[5]), 4)
        
        #Vect methods
        estimated = p.vect_prob_tag(range(6)).round(4)
        expected = tag_pager.round(4)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        expected = np.log2(tag_pager).round(4)
        estimated = p.vect_log_prob_tag(range(6)).round(4)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

    def test_tag_given_item(self):
        self.__init_test(test.SMALL_DEL_FILE)
        p = RWEstimator(self.annots)
        
        ntags, nitems, edges = graph.iedge_from_annotations(self.annots)
        nx_graph = graph.create_nxgraph(edges)
        
        pers_vect = dict((node_id, 0) for node_id in xrange(ntags + nitems))
        probs = []
        log_probs = []
        for tag in xrange(6):
            pers_vect[tag] = 1
            pagerank_dict = nx.pagerank(nx_graph, personalization = pers_vect)
            pageranks = [pagerank_dict[node_id] for node_id in sorted(pagerank_dict)]
        
            item_given_tag_pager = np.array(pageranks[ntags:])
            item_given_tag_pager /= item_given_tag_pager.sum()
            
            prob_tag = p.prob_tag(tag)
            prob_item = p.prob_item(0)
            tag_given_item_pager = item_given_tag_pager * prob_tag / prob_item
        
            expected = tag_given_item_pager[0]
            computed = p.prob_tag_given_item(0, tag)
            self.assertAlmostEquals(computed, expected, 4)

            log_expected = np.log2(tag_given_item_pager[0])
            log_computed = p.log_prob_tag_given_item(0, tag)
            self.assertAlmostEquals(log_computed, log_expected, 4)
            
            probs.append(expected)
            log_probs.append(log_expected)
            pers_vect[tag] = 0
    
        #Vect methods
        estimated = p.vect_prob_tag_given_item(0, range(6)).round(4)
        expected = np.array(probs).round(4)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())

        #Log vect methods
        estimated = p.vect_log_prob_tag_given_item(0, range(6)).round(3)
        expected = np.array(log_probs).round(3)
        self.assertTrue(np.in1d(expected, estimated).all())
        self.assertTrue(np.in1d(estimated, expected).all())
    
    def test_prob_user(self):
        self.__init_test(test.SMALL_DEL_FILE)
        p = RWEstimator(self.annots)
        
        #User and user given item
        prob = p.prob_user(0)
        expected_prob_u = p.prob_tag(0) * p.prob_tag(1) * p.prob_tag(2) 
        self.assertEquals(prob, expected_prob_u)
        
        prob = p.log_prob_user(0)
        self.assertAlmostEquals(prob, np.log2(expected_prob_u))

        prob = p.prob_user_given_item(0, 0)
        expected_prob_ugi = p.prob_tag_given_item(0, 0) * p.prob_tag_given_item(0, 1) * p.prob_tag_given_item(0, 2) 
        self.assertEquals(prob, expected_prob_ugi)
        
        prob = p.log_prob_user_given_item(0, 0)
        self.assertAlmostEquals(prob, np.log2(expected_prob_ugi))
        
if __name__ == "__main__":
    unittest.main()