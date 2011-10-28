# -*- coding: utf8

from __future__ import division, print_function

from tagassess import data_parser
from tagassess import tagcloud
from tagassess import test

import unittest

class TestPreComputedValuesCloud(unittest.TestCase):
    
    def setUp(self):
        self.annots = []
        parser = data_parser.Parser()
        with open(test.SMALL_DEL_FILE) as in_f:
            for annot in parser.iparse(in_f, 
                                       data_parser.delicious_flickr_parser):
                self.annots.append(annot)
                    
    def tearDown(self):
        self.annots = None
    
    def test_creation(self):
        tag_value_map = {0:2,
                         1:1,
                         2:2,
                         3:0,
                         4:3,
                         5:0}
        
        cloud = tagcloud.PreComputedValuesCloud(self.annots, tag_value_map, 
                                                cloud_size = 3)
        self.assertEqual(set([4, 0, 2]), cloud.current_cloud)
        self.assertEqual(3, len(cloud))
        
        cloud.update([0])
        self.assertEqual(set([4, 0, 1]), cloud.current_cloud)
        
        cloud.update([0, 1, 3])
        self.assertEqual(set([0, 1, 4]), cloud.current_cloud)
        
        cloud.update([0, 1, 5])
        self.assertEqual(set([]), cloud.current_cloud)
            
    def test_coverage(self):
        
        tag_value_map = {0:2,
                         1:1,
                         2:2,
                         3:0,
                         4:3,
                         5:0}
        
        cloud = tagcloud.PreComputedValuesCloud(self.annots, tag_value_map, 
                                                cloud_size = 3)
        
        self.assertEqual(0, cloud.coverage([7, 8, 9]))
        self.assertEqual(2 / 3, cloud.coverage([0, 2, 5]))
        self.assertEqual(1, cloud.coverage([0, 4, 2]))
        
        cloud.update([0])
        self.assertEqual(0, cloud.coverage([7, 8, 9]))
        self.assertEqual(2 / 3, cloud.coverage([0, 1, 5]))
        self.assertEqual(1, cloud.coverage([0, 4, 1]))
        
    def test_recall(self):
        tag_value_map = {0:2,
                         1:1,
                         2:2,
                         3:0,
                         4:3,
                         5:0}
        
        cloud = tagcloud.PreComputedValuesCloud(self.annots, tag_value_map, 
                                                cloud_size = 3)
        
        self.assertEqual(0, cloud.recall([7, 8, 9]))
        self.assertEqual(0.5, cloud.recall([0, 1, 3, 4]))
        self.assertEqual(1, cloud.recall([0, 1]))
        self.assertEqual(1, cloud.recall([0, 1, 2]))

        cloud.update([0])
        self.assertEqual(0, cloud.recall([7, 8, 9]))
        self.assertEqual(3 / 4, cloud.recall([0, 1, 3, 4]))

    def test_query_recall(self):
        tag_value_map = {0:2,
                         1:1,
                         2:2,
                         3:0,
                         4:3,
                         5:0}
        
        cloud = tagcloud.PreComputedValuesCloud(self.annots, tag_value_map, 
                               cloud_size = 3)
        cloud.update([1])
        self.assertEqual(0, cloud.query_recall([7, 8, 9]))
        self.assertEqual(2 / 3, cloud.query_recall([0, 1, 4]))

    def test_precision(self):
        tag_value_map = {0:2,
                         1:1,
                         2:2,
                         3:0,
                         4:3,
                         5:0}
        
        cloud = tagcloud.PreComputedValuesCloud(self.annots, tag_value_map, 
                                                cloud_size = 3)
        
        self.assertEqual(0, cloud.precision([7, 8, 9]))
        self.assertEqual(2 / 3, cloud.precision([0, 1, 3, 4]))
        self.assertEqual(1 / 3, cloud.precision([0]))
        self.assertEqual(1, cloud.precision([0, 1, 2]))


        cloud.update([0])
        self.assertEqual(0, cloud.recall([7, 8, 9]))
        self.assertEqual(3 / 5, cloud.recall([0, 1, 3, 4, 5]))
        self.assertEqual(4 / 5, cloud.recall([0, 2, 3, 4, 5]))
        
    def test_query_precision(self):
        tag_value_map = {0:2,
                         1:1,
                         2:2,
                         3:0,
                         4:3,
                         5:0}
        
        cloud = tagcloud.PreComputedValuesCloud(self.annots, tag_value_map, 
                                                cloud_size = 3)
        cloud.update([1])
        self.assertEqual(0, cloud.query_precision([7, 8, 9]))
        self.assertEqual(0.75, cloud.precision([0, 1, 3, 4]))

class TestTFIDFCloud(unittest.TestCase):
    
    def setUp(self):
        self.annots = []
        parser = data_parser.Parser()
        with open(test.SMALL_DEL_FILE) as in_f:
            for annot in parser.iparse(in_f, 
                                       data_parser.delicious_flickr_parser):
                self.annots.append(annot)
                    
    def tearDown(self):
        self.annots = None
    
    def test_creation_tf(self):
        cloud = tagcloud.TFIDFCloud(self.annots, 3, 'tf')
        self.assertEqual(set([0, 1, 2]), cloud.current_cloud)
        self.assertEqual(3, len(cloud))
        
        cloud.update([0]) #Possible tags [0, 1, 3, 4, 5]
        self.assertEqual(set([0, 1, 3]), cloud.current_cloud)
    
    def test_creation_idf(self):
        cloud = tagcloud.TFIDFCloud(self.annots, 3, 'idf')
        self.assertEqual(set([2, 3, 4]), cloud.current_cloud)
        self.assertEqual(3, len(cloud))
        
        cloud.update([0]) #Possible tags [0, 1, 3, 4, 5]
        self.assertEqual(set([3, 4, 5]), cloud.current_cloud)
    
    def test_creation_tfidf(self):
        cloud = tagcloud.TFIDFCloud(self.annots, 3, 'tf-idf')
        self.assertEqual(set([0, 1, 2]), cloud.current_cloud)
        self.assertEqual(3, len(cloud))
        
        cloud.update([0]) #Possible tags [0, 1, 3, 4, 5]
        self.assertEqual(set([4, 0, 3]), cloud.current_cloud)
        
    def test_creation_invidf(self):
        cloud = tagcloud.TFIDFCloud(self.annots, 3, 'inv-idf')
        self.assertEqual(set([0, 1, 2]), cloud.current_cloud)
        self.assertEqual(3, len(cloud))
        
        cloud.update([0]) #Possible tags [0, 1, 3, 4, 5]
        self.assertEqual(set([0, 1, 3]), cloud.current_cloud)