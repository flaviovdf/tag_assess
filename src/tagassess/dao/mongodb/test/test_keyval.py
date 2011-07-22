# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=C0103

from __future__ import print_function, division

from tagassess.dao.mongodb import MongoDBException
from tagassess.dao.mongodb.keyval import KeyValStore
from tagassess.dao.mongodb.test import MongoManager, PORT

import unittest

class TestKeyVal(unittest.TestCase):
    
    def setUp(self):
        self.manager = MongoManager()
        self.manager.start_mongo()
        
    def tearDown(self):
        if self.manager:
            self.manager.stop_mongo()
    
    def test_get_all(self):
        with KeyValStore('test', connection_port = PORT) as keyval:
            keyval.create_table('bah')
            keyval[1] = 'a'
            keyval[2] = 'b'
            cache = keyval.get_all()
            
            self.assertEquals(2, len(cache))
            self.assertEquals('a', cache[1])
            self.assertEquals('b', cache[2])
    
    def test_all(self):
        with KeyValStore('test', connection_port = PORT) as keyval:
            keyval.create_table('bah')
            self.assertEquals(0, len(keyval))
            
            keyval[1] = 2
            self.assertTrue(1 in keyval)
            self.assertEquals(2, keyval[1])
            self.assertEquals(1, len(keyval))
            
            self.assertTrue(0 not in keyval)
            try:
                x = keyval[0]
                self.fail()
            except KeyError:
                pass
        
            try:
                keyval[1] = 3
                self.fail()
            except MongoDBException:
                pass
            
            keyval[(2, 3)] = 'a'
            keyval[4] = (5, 6)
            
            self.assertEquals('a', keyval[(2, 3)])
            self.assertEquals('a', keyval[[2, 3]])
            self.assertEquals([5, 6], keyval[4])
            
            count = 0
            for key, val in keyval:
                count += 1
            self.assertEquals(3, count)