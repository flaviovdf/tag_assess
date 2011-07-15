# -*- coding: utf8
'''
MongoDB interface for accessing annotations.
'''
from __future__ import division, print_function

from tagassess.dao.base import Base
from pymongo import Connection

class MongoDBException(Exception):
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
            raise MongoDBException('Collection does not exist')
        
        self.table = self.database[tname]