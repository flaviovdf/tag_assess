# -*- coding: utf8
#pylint: disable-msg=W0401
#pylint: disable-msg=W0614
'''Classes for reading `IndexInf` objects from PyTables'''

from __future__ import division, print_function

from tagassess.dao.base import BaseDao
from tagassess.dao.base import BaseReader
from tagassess.dao.base import BasePyTablesWrapper
from tables import *

class IndexReader(BaseReader):
    '''
    A `IndexReader` is used to read PyTables H5 files and convert
    data to `IndexInf`.
    '''
    def __init__(self, fpath):
        super(IndexReader, self).__init__(fpath, 'r')
        self.conv_func = lambda row: IndexInf(row['POSTERIOR'], row['TAG'],
                                              row['LFREQ'], row['GFREQ'])
    
    def get_conversion_func(self):
        '''Returns a callable which converts rows to `IndexInf`.'''
        return self.conv_func
    
class IndexWriter(BasePyTablesWrapper):

    '''
    A `IndexWriter` is used to create and write `IndexInf` 
    objects to PyTables H5 files.
    '''

    def __init__(self, fpath, mode='a'):
        super(IndexWriter, self).__init__(fpath, mode)
        self.table = None

    def create_table(self, table_name):
        '''Creates a new table with the given name.'''
        self.table = super(IndexWriter, self)._create_table(table_name,
                                                            IndexDesc)

    def write(self, index_inf):
        '''
        Writes a new index to the *current* position of the
        table.

        Arguments
        ---------
        index_inf: an `IndexInf`
        '''
        self.table.row['POSTERIOR'] =  index_inf.get_posterior()
        self.table.row['TAG'] =  index_inf.get_tag()
        self.table.row['LFREQ'] =  index_inf.get_local_frequency()
        self.table.row['GFREQ'] =  index_inf.get_global_frequency()
        self.table.row.append()

class IndexInf(BaseDao):
    '''Represents information stored at index files'''
    
    def __init__(self, posterior, tag, local_frequency, global_frequency):
        super(IndexInf, self).__init__()
        self.__post = posterior
        self.__tag = tag
        self.__local_frequency = local_frequency
        self.__global_frequency = global_frequency
        
        #Used for hashing and equality
        self.__tuple = (('POSTERIOR', posterior), ('TAG', tag))
    
    def get_posterior(self):
        '''Return posterior as int'''
        return self.__post

    def get_tag(self):
        '''Return tag as int'''
        return self.__tag

    def get_local_frequency(self):
        '''Return item as int'''
        return self.__local_frequency

    def get_global_frequency(self):
        '''Return item as int'''
        return self.__global_frequency

    def get_tuple(self):
        '''Return tuple which represents annotation'''
        return self.__tuple

class IndexDesc(IsDescription):
    '''
    Defines an index description to be saved on file.
    '''

    POSTERIOR = Int32Col()   #@UndefinedVariable
    TAG       = Int32Col()   #@UndefinedVariable
    LFREQ     = Int32Col()   #@UndefinedVariable
    GFREQ     = Int32Col()   #@UndefinedVariable