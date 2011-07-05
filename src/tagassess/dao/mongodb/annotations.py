# -*- coding: utf8
'''Classes for reading annotations from MongoDB'''
from __future__ import division, print_function

from tagassess.dao.base import Base
from tagassess.dao.base import Reader
from tagassess.dao.base import Writer
from pymongo import Connection, ASCENDING

FIELDS = {u'user':1, u'item':1, u'tag':1, u'date':1, '_id':0}

class MongoDBAnnotationsException(Exception):
    '''Signals errors when dealing with mongodb'''
    pass

class BaseMongo(Base):
    '''
    Contains base open and close connection methods
    for MongoDB
    '''
    def __init__(self, database_name, connection_host=None, 
                 connection_port=None):
        super(BaseMongo, self).__init__()
        self.connection_host = connection_host
        self.connection_port = connection_port
        self.database_name = database_name
        self.opened = False
        self.connection = None
        self.database = None
        self.table = None
        
    def open(self):
        if not self.opened:
            self.connection = Connection(self.connection_host, 
                                         self.connection_port)
            self.database = self.connection[self.database_name]
            self.opened = True
            self.table = None

    def close(self):
        if self.opened:
            self.connection.disconnect()
            self.database = None
            self.connection = None
            self.opened = False
            self.table = None

    def change_table(self, tname, **kwargs):
        if tname not in self.database.collection_names():
            raise MongoDBAnnotationsException('Collection does not exist')
        
        self.table = self.database[tname]

class AnnotReader(BaseMongo, Reader):
    '''
    A `AnnotReader` is used to read annotation from a mongodb.
    '''
    def __init__(self, database_name, connection_host=None, 
                 connection_port=None):
        super(AnnotReader, self).__init__(database_name)
        
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
        super(AnnotWriter, self).__init__(database_name)

    def create_table(self, tname, **kwargs):
        if tname in self.database.collection_names():
            raise MongoDBAnnotationsException('Collection exists')
        
        self.table = self.database[tname]
        self.table.ensure_index([("user", ASCENDING), ("item", ASCENDING), 
                                 ("tag", ASCENDING)])

    def append_row(self, row, **kwargs):
        self.table.insert(row)