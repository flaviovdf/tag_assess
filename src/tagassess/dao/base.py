# -*- coding: utf8
'''
Contains base classes for accessing data stored on DBs.
'''
from __future__ import division, print_function

import abc

class Base(object):
    '''
    Base wrapper for accessing DBs
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def open(self, **kwargs):
        '''Opens connection to database'''
        pass
    
    @abc.abstractmethod
    def close(self, **kwargs):
        '''Closes connection to database'''
        pass

    @abc.abstractmethod
    def change_table(self, tname, **kwargs):
        '''
        Changes to another table that can be written or
        read from.
        
        Arguments
        ---------
        tname: str
            The name of the table
        '''
        pass

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, type, value, traceback):
        '''Closes the file return `True` if no exception was caught'''
        self.close()
        #Value is an exception in case with fails.
        return not isinstance(value, Exception) 

class Reader(Base):
    '''
    Defines base methods for DB readers.
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def iterate(self, query = None, **kwargs):
        '''
        Returns a iterator of the table under the given conditions.
        
        Arguments
        ---------
        query (optional): any (depends on subclass)
            A query to filter some rows
            
        Returns
        -------
        Each element of the iterator should be a dict
        '''
        pass
    
class Writer(Base):
    '''
    Defines base methods for DB writers.
    '''
    __metaclass__ = abc.ABCMeta
    
    @abc.abstractmethod
    def append_row(self, row, **kwargs):
        '''
        Appends a new row to the end of the table.
        
        Arguments
        ---------
        row: dict
            The row to append
        '''
        pass

    @abc.abstractmethod
    def create_table(self, tname, **kwargs):
        '''
        Creates a new table. Returns
        the table if creation was successful, `None` otherwise.

        Arguments
        ---------
        tname: str
            The name of the new table
        '''
        pass