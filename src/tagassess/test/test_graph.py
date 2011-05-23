# -*- coding: utf8
#pylint: disable-msg=C0103
#pylint: disable-msg=C0111
#pylint: disable-msg=C0301
#pylint: disable-msg=E1101

from __future__ import division, print_function

from tagassess import data_parser
from tagassess import graph 
from tagassess import test

from tagassess.dao import annotations

import os
import tempfile
import unittest

class TestGraph(unittest.TestCase):
    
    def setUp(self):
        self.h5_file = tempfile.mktemp('testw.h5')
        parser = data_parser.Parser()
        with open(test.SMALL_DEL_FILE) as in_f:
            with annotations.AnnotWriter(self.h5_file) as writer:
                writer.create_table('deli')
                for annot in parser.iparse(in_f, data_parser.delicious_flickr_parser):
                    writer.write(annot)
                    
    def tearDown(self):
        os.remove(self.h5_file)
        
    def test_edge_list(self):
        ntags, nsinks, iedges = graph.iedge_list(self.h5_file, 'deli')
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
        edges = [e for e in graph.iedge_list(self.h5_file, 'deli')[2]]
        g = graph.create_igraph(edges)
        
        paths = g.shortest_paths_dijkstra([0])
        inf = float('inf')
        self.assertEquals(paths, [[0, 1, inf, 1, 1, 1, 1, inf, 1, 2, 2]])
        
if __name__ == "__main__":
    unittest.main()