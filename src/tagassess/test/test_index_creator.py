# -*- coding: utf8
from __future__ import division, print_function

from collections import defaultdict

from tagassess import data_parser
from tagassess import test
from tagassess.index_creator import create_double_occurrence_index
from tagassess.index_creator import create_occurrence_index
from tagassess.index_creator import create_metrics_index

import random
import time
import unittest

class TestIndexCreation(unittest.TestCase):

    def _base_metrics(self, use_user):
        '''Creates and counts the popularity of random tags. The
        test will compare this with the result of the indices.'''
        
        #Generating some random annotations
        any_date = time.time()
        tag_pop = defaultdict(int)
        post_tag_pop = defaultdict(lambda: defaultdict(int))
        
        annotations = []
        for i in xrange(32):
            user = random.randint(0, 4)
            tag = random.randint(0, 4)
            item = random.randint(0, 4)
            annotations.append(data_parser.to_json(user, item, tag, any_date))
            
            post = user if use_user else item
            
            tag_pop[tag] += 1
            post_tag_pop[post][tag] += 1
        
        if use_user:
            index = create_metrics_index(annotations, 'user', 'tag')
        else:
            index = create_metrics_index(annotations, 'item', 'tag')
            
        self.assertEquals(post_tag_pop, index[0])
        self.assertEquals(tag_pop, index[2])
            
    def test_metrics_index(self):
        self._base_metrics(False)
        self._base_metrics(True)

    def test_metrics_small_file(self):
        p = data_parser.Parser()
        with open(test.SMALL_DEL_FILE) as f:
            annots = [a for a in p.iparse(f, data_parser.delicious_flickr_parser)]
            
        item_tag_frequencies, collection_item_frequency, \
            collection_tag_frequency1 = create_metrics_index(annots, 'item', 'tag')
            
        user_tag_frequencies, collection_user_frequency, \
            collection_tag_frequency2 = create_metrics_index(annots, 'user', 'tag')
        
        self.assertEquals(collection_item_frequency[0], 5)
        self.assertEquals(collection_item_frequency[1], 1)
        self.assertEquals(collection_item_frequency[2], 2)
        self.assertEquals(collection_item_frequency[3], 1)
        self.assertEquals(collection_item_frequency[4], 1)
        
        self.assertEquals(collection_user_frequency[0], 4)
        self.assertEquals(collection_user_frequency[1], 4)
        self.assertEquals(collection_user_frequency[2], 2)
        
        self.assertEquals(item_tag_frequencies[0][0], 2)
        self.assertEquals(item_tag_frequencies[0][1], 1)
        self.assertEquals(item_tag_frequencies[0][2], 0)
        self.assertEquals(item_tag_frequencies[0][3], 1)
        self.assertEquals(item_tag_frequencies[0][4], 1)
        self.assertEquals(item_tag_frequencies[0][5], 0)
        
        self.assertEquals(user_tag_frequencies[2][0], 0)
        self.assertEquals(user_tag_frequencies[2][1], 0)
        self.assertEquals(user_tag_frequencies[2][2], 0)
        self.assertEquals(user_tag_frequencies[2][3], 0)
        self.assertEquals(user_tag_frequencies[2][4], 1)
        self.assertEquals(user_tag_frequencies[2][5], 1)
        
        self.assertEquals(collection_tag_frequency1[0], 3)
        self.assertEquals(collection_tag_frequency1[1], 3)
        self.assertEquals(collection_tag_frequency1[2], 1)
        self.assertEquals(collection_tag_frequency1[3], 1)
        self.assertEquals(collection_tag_frequency1[4], 1)
        self.assertEquals(collection_tag_frequency1[5], 1)
        self.assertEquals(collection_tag_frequency1, collection_tag_frequency2)
            

    def test_occurence_index_user_to_item(self):
        #Not the best of names, but we attribute this to fields
        #which have no impact on the test.
        no_impact = 1
        
        a1 = data_parser.to_json(1, 1, no_impact, no_impact)
        a2 = data_parser.to_json(1, 2, no_impact, no_impact)
        a3 = data_parser.to_json(1, 1, no_impact, no_impact)
        a4 = data_parser.to_json(2, 2, no_impact, no_impact)
        a5 = data_parser.to_json(2, 3, no_impact, no_impact)
    
        index = create_occurrence_index([a1, a2, a3, a4, a5], 'user', 'item')
        self.assertEqual(index[1], set([1, 2, 1]))
        self.assertEqual(index[2], set([2, 3]))
    
    def test_occurence_index_user_to_tag(self):
        #Not the best of names, but we attribute this to fields
        #which have no impact on the test.
        no_impact = 1
        
        a1 = data_parser.to_json(1, no_impact, 1, no_impact)
        a2 = data_parser.to_json(1, no_impact, 2, no_impact)
        a3 = data_parser.to_json(1, no_impact, 1, no_impact)
        a4 = data_parser.to_json(2, no_impact, 2, no_impact)
        a5 = data_parser.to_json(2, no_impact, 3, no_impact)
    
        index = create_occurrence_index([a1, a2, a3, a4, a5], 'user', 'tag')
        self.assertEqual(index[1], set([1, 2, 1]))
        self.assertEqual(index[2], set([2, 3]))
    
    def test_occurence_index_item_to_user(self):
        #Not the best of names, but we attribute this to fields
        #which have no impact on the test.
        no_impact = 1
        
        a1 = data_parser.to_json(1, 1, no_impact, no_impact)
        a2 = data_parser.to_json(1, 2, no_impact, no_impact)
        a3 = data_parser.to_json(1, 1, no_impact, no_impact)
        a4 = data_parser.to_json(2, 2, no_impact, no_impact)
        a5 = data_parser.to_json(2, 3, no_impact, no_impact)
            
        index = create_occurrence_index([a1, a2, a3, a4, a5], 'item', 'user')
        self.assertEqual(index[1], set([1, 1]))
        self.assertEqual(index[2], set([1, 2]))
        self.assertEqual(index[3], set([2]))
        
    def test_double_occurrence_index(self):
        no_impact = 1
        
        a1 = data_parser.to_json(1, no_impact, 1, no_impact)
        a2 = data_parser.to_json(1, no_impact, 2, no_impact)
        a3 = data_parser.to_json(1, no_impact, 1, no_impact)
        a4 = data_parser.to_json(2, no_impact, 2, no_impact)
        a5 = data_parser.to_json(2, no_impact, 3, no_impact)
    
        from_to, inv = create_double_occurrence_index([a1, a2, a3, a4, a5], 
                                                      'user', 'tag')
        self.assertEqual(from_to[1], set([1, 2, 1]))
        self.assertEqual(from_to[2], set([2, 3]))
        
        self.assertEqual(inv[1], set([1]))
        self.assertEqual(inv[2], set([1, 2]))
        self.assertEqual(inv[3], set([2]))
        
if __name__ == "__main__":
    unittest.main()