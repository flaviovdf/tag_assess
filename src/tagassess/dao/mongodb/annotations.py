# -*- coding: utf8
'''Classes for reading annotations from MongoDB'''
from __future__ import division, print_function

from pymongo import ASCENDING
from tagassess.dao.base import Reader
from tagassess.dao.base import Writer
from tagassess.dao.mongodb import BaseMongo, MongoDBException

FIELDS = {u'user':1, u'item':1, u'tag':1, u'date':1, '_id':0}

class AnnotReader(BaseMongo, Reader):
    '''
    A `AnnotReader` is used to read annotation from a mongodb.
    '''
    def __init__(self, database_name, connection_host=None, 
                 connection_port=None):
        super(AnnotReader, self).__init__(database_name,
                                          connection_host=connection_host,
                                          connection_port=connection_port)
        
    def iterate(self, query = None, **kwargs):
        iterable = None
        if query:
            iterable = self.table.find(query, fields = FIELDS)
        else:
            iterable = self.table.find(fields = FIELDS)
        
        return iterable
    
class AnnotWriter(BaseMongo, Writer):
    '''
    A `AnnotWriter` is used to write annotations 
    to mongo databases.
    '''
    def __init__(self, database_name, connection_host=None, 
                 connection_port=None):
        super(AnnotWriter, self).__init__(database_name,
                                          connection_host=connection_host,
                                          connection_port=connection_port)

    def create_table(self, tname, **kwargs):
        if tname in self.database.collection_names():
            raise MongoDBException('Collection exists')
        
        self.table = self.database[tname]
        self.table.ensure_index([("user", ASCENDING), ("item", ASCENDING), 
                                 ("tag", ASCENDING)])

    def append_row(self, row, **kwargs):
        self.table.insert(row)