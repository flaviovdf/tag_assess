# -*- coding: utf8
'''Module with functions for accessing key value pair in MongoDB'''
from __future__ import division, print_function

from pymongo import ASCENDING
from tagassess.dao.mongodb import BaseMongo, MongoDBException

FIELDS = {u'key':1, u'value':1, '_id':0}

class KeyValStore(BaseMongo):
    '''
    A `KeyValStore` is used to store key value pairs in a MongoDB
    '''
    def __init__(self, database_name, connection_host=None, 
                 connection_port=None):
        super(KeyValStore, self).__init__(database_name)
    
    def __getitem__(self, key):
        return self.get(key)
    
    def __setitem__(self, key, val):
        self.put(key, val)
    
    def __contains__(self, key):
        return self.has_key(key)

    def __iter__(self):
        def generator():
            '''Yields key, value pairs'''
            for pair in self.table.find(fields = FIELDS):
                yield pair['key'], pair['value']
        return generator()
    
    def __len__(self):
        return self.table.count()
    
    def put(self, key, val, no_check = False):
        '''
        Associates the given value to the key. `MongoDBException` is
        thrown if a key already exists.
        
        Arguments
        ---------
        key: any
            The key to lookup
        val: any
            Value associate with
        no_check: bool (optional, defaults to False)
            Use this to override the verification if the key already
            exists. This will lead to inconsistency if the key is there, 
            so use only when you need the runtime improvement.
        '''
        if (not no_check) and self.has_key(key):
            raise MongoDBException('Key already in store')
        self.table.insert({'key':key, 'value':val})
    
    def get(self, key):
        '''
        Returns the value associated with the given key.
        Throws `KeyError` if no value is found.
        
        Arguments
        ---------
        key: any
            The key to lookup
        '''
        val = self.table.find_one({u'key':key}, fields = FIELDS)
        if val is None:
            raise KeyError()
        else:
            return val['value']
    
    def has_key(self, key):
        '''
        Tests if key has been inserted to `KeyValStore`
        
        Arguments
        ---------
        key: any
            The key to test
        '''
        return self.table.find_one({u'key':key}) != None
    
    def create_table(self, tname):
        '''
        Creates a new key value table and changes to it
        
        Arguments
        ---------
        tname: str
            Name of the new table
        '''
        if tname in self.database.collection_names():
            raise MongoDBException('Collection exists')
        
        self.table = self.database[tname]
        self.table.ensure_index([('key', ASCENDING)])
    
    def get_all(self):
        '''
        Populates a dictionary with all key, val pairs.
        '''
        return dict((key, val) for key, val in self)