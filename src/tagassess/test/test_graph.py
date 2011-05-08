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
        
    def test_extract_index(self):
        index = graph.extract_indexes_from_file(self.h5_file, 'deli')[0]
        expect = {0: set([0, 1, 3, 4]), 
                  1: set([2]), 
                  2: set([0, 5]), 
                  3: set([1]), 
                  4: set([1])}
        self.assertEquals(index, expect)
        
    def test_edge_list(self):
        base_index, tag_to_item = \
            graph.extract_indexes_from_file(self.h5_file, 'deli')
        edges = graph.edge_list(base_index, tag_to_item, False)[1]

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
        
        self.assertEquals(set(edges), set(expected))
    
    def test_graph(self):
        base_index, tag_to_item = \
            graph.extract_indexes_from_file(self.h5_file, 'deli')
        g = graph.create_igraph(base_index, tag_to_item, False)
        
        paths = g.shortest_paths([0])
        inf = float('inf')
        self.assertEquals(paths, [[0, 1, inf, 1, 1, 1, 1, inf, 1, 2, 2]])
        
if __name__ == "__main__":
    unittest.main()