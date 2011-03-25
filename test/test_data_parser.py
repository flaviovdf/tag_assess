# -*- coding: utf8

from __future__ import print_function, division

import data_parser
import time
import unittest

DELICIOUS_LINE1 = '2003-01-01 01:00:00     2384    125497  tinker'
DELICIOUS_LINE2 = '2011-02-17 11:10:20     2384    674518  hardware'
DELICIOUS_LINE3 = '2003-01-01 01:00:00     1       674518  hardware pc'

class TestAll(unittest.TestCase):
    
    def test_delicious_flickr(self):
        user, item, tags, date = data_parser.delicious_flickr_parser(DELICIOUS_LINE1)
        
        self.assertEqual(2384, user)
        self.assertEqual(125497, item)
        self.assertEqual(['tinker'], tags)
        self.assertEqual('2003-01-01 01:00:00', time.strftime('%Y-%m-%d %H:%M:%S', date))
        
        user, item, tags, date = data_parser.delicious_flickr_parser(DELICIOUS_LINE2)
        self.assertEqual(2384, user)
        self.assertEqual(674518, item)
        self.assertEqual(['hardware'], tags)
        self.assertEqual('2011-02-17 11:10:20', time.strftime('%Y-%m-%d %H:%M:%S', date))
        
        user, item, tags, date = data_parser.delicious_flickr_parser(DELICIOUS_LINE3)
        self.assertEqual(1, user)
        self.assertEqual(674518, item)
        self.assertEqual(['hardware', 'pc'], tags)
        self.assertEqual('2003-01-01 01:00:00', time.strftime('%Y-%m-%d %H:%M:%S', date))