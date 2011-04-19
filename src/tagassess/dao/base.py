# -*- coding: utf8
#pylint: disable-msg=W0401
#pylint: disable-msg=W0614
#pylint: disable-msg=W0622

'''
Contains base classes for accessing data stored on PyTables files.
'''
from __future__ import division, print_function

from tables import * #Wildcard necessary for PyTables.

import abc
import itertools

class BasePyTablesWrapper(object):
    '''
    Base wrapper for accessing PyTables. Defines basic functions
    for creating, populating and searching PyTables databases. This
    class is to be extended by others so that table rows can be converted
    to and from dao objects.
    '''
    
    def __init__(self, fpath, mode):
        self.fpath = fpath
        self.mode = mode
        self.opened = False
        self.tablefile = None
        
    def open_file(self):
        '''Opens the file given in the constructor.'''
        if not self.opened:
            self.tablefile = openFile(self.fpath, self.mode)
            self.opened = True

    def close_file(self):
        '''Closes the file. If the file is already closed, does nothing.'''
        if self.opened:
            self.tablefile.close()
            self.opened = False

    def _get_table(self, table_name):
        '''
        Jumps to a table returning it. 
        '''
        return self.tablefile.getNode('/', table_name)

    def _create_table(self, tname, desc):
        '''
        Creates a new table in the annotation H5 file. Returns
        the table if creation was successful, `None` otherwise.

        Arguments
        ---------
        tname: str
            The name of the new table
        desc: IsDescription
            Describes the table
        '''
        if self.opened:
            return self.tablefile.createTable(self.tablefile.root,
                                              tname, desc)

    def _iterate(self, table_name, conv_function, condition=None):
        '''
        Returns a iterator of the table under the given conditions.
        
        Arguments
        ---------
        table_name: str
            The table to iterate over
        conv_function: callable
            The function which will convert row to objects
        condition:
            A condition to filter some rows such as: where date > x.
        '''
        table = self._get_table(table_name)

        iterable = None
        if condition:
            iterable = table.where(condition)
        else:
            iterable = table
            
        return itertools.imap(conv_function, iterable)

    def __enter__(self):
        self.open_file()
        return self

    def __exit__(self, type, value, traceback):
        '''Closes the file return `True` if no exception was caught'''
        self.close_file()
        return not value #Value is an exception in case with fails.

class BaseReader(BasePyTablesWrapper):
    '''Base class for creating table reader'''
    
    def iterate(self, table_name, condition=None):
        '''
        Returns a iterator of the table under the given conditions.
        This iterator yields objects according to the `get_base_func`
        method.
        
        Arguments
        ---------
        table_name: str
            The table to iterate over
        condition:
            A condition to filter some rows such as: where date > x.
        '''
        return super(BaseReader, self)._iterate(table_name, 
                                                self.get_conversion_func(), 
                                                condition)
    
    @abc.abstractmethod
    def get_conversion_func(self):
        '''Gets the callable which will convert table rows to objects.'''
        pass

class BaseDao(object):
    '''Base dao which defines equality and hash'''
    
    @abc.abstractmethod
    def get_tuple(self):
        '''Return the tuple representation of the dao'''
        pass

    def __eq__(self, other):
        return isinstance(other, BaseDao) and \
               self.get_tuple() == other.get_tuple()

    def __hash__(self):
        return hash(self.get_tuple())