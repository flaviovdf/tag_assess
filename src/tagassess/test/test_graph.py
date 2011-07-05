# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=E1101

from __future__ import division, print_function

from tagassess import data_parser
from tagassess import graph 
from tagassess import test

import networkx as nx
import unittest

class TestGraph(unittest.TestCase):
    
    def setUp(self):
        self.annots = []
        parser = data_parser.Parser()
        with open(test.SMALL_DEL_FILE) as in_f:
            for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                self.annots.append(annot)
                    
    def tearDown(self):
        self.annots = None
        
    def test_edge_list(self):
        ntags, nsinks, iedges = graph.iedge_from_annotations(self.annots)
        self.assertEqual(6, ntags)
        self.assertEqual(5, nsinks)
        
        edges = set(e for e in iedges)

        outgo_edges = [(0, 1),
                       (0, 3),
                       (0, 4),
                       (0, 5),
                       (1, 3),
                       (1, 4),
                       (4, 3)]
        #Inverse edges
        expected = [] + outgo_edges #Copy
        expected.extend((v, k) for k, v in outgo_edges)
        
        #Edges to items
        expected.extend([(0, 6),
                         (0, 8),
                         (1, 6),
                         (1, 9),
                         (1, 10),
                         (2, 7),
                         (3, 6),
                         (4, 6),
                         (5, 8)])
        
        self.assertEquals(edges, set(expected))
    
    def test_graph(self):
        edges = [e for e in graph.iedge_from_annotations(self.annots)[2]]
        g = graph.create_nxgraph(edges)
        
        paths = nx.shortest_path_length(g, source = 0)
        self.assertEquals(paths, {0: 0, 1: 1, 3: 1, 4: 1, 5: 1, 6: 1, 8: 1, 9: 2, 10: 2})
        
if __name__ == "__main__":
    unittest.main()