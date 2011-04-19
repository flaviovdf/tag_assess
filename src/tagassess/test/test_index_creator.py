# -*- coding: utf8
#pylint: disable-msg=C0111
#pylint: disable-msg=R0201
#pylint: disable-msg=W0401
#pylint: disable-msg=W0612
#pylint: disable-msg=W0614

from __future__ import division, print_function

from collections import defaultdict
from tagassess.dao.annotations import Annotation
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
            annotations.append(Annotation(user, item, tag, any_date))
            
            post = user if use_user else item
            
            tag_pop[tag] += 1
            post_tag_pop[post][tag] += 1
        
        index = create_metrics_index(annotations, use_user)
        for i in index:
            post = i.get_posterior()
            tag = i.get_tag()
            self.assertEquals(tag_pop[tag], i.get_global_frequency())
            self.assertEquals(post_tag_pop[post][tag], i.get_local_frequency())
            
    def test_metrics_index(self):
        self._base_metrics(False)
        self._base_metrics(True)
                 
if __name__ == "__main__":
    unittest.main()