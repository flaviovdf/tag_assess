# -*- coding: utf8
#pylint: disable-msg=W0401
#pylint: disable-msg=W0614
'''Classes for reading `Annotation` objects from PyTables'''

from __future__ import division, print_function

from tagassess.dao.base import BaseDao
from tagassess.dao.base import BaseReader
from tagassess.dao.base import BasePyTablesWrapper
from tables import *

class AnnotReader(BaseReader):
    '''
    A `AnnotReader` is used to read PyTables H5 files and convert
    data to `Annotations`.
    '''

    def __init__(self, fpath):
        super(AnnotReader, self).__init__(fpath, 'r')
        self.conv_func = lambda row: Annotation(row['USER'], row['ITEM'],
                                                row['TAG'], row['DATE'])
    
    def get_conversion_func(self):
        '''Returns a callable which converts rows to `Annotation`.'''
        return self.conv_func


class AnnotWriter(BasePyTablesWrapper):

    '''
    A `AnnotWriter` is used to create and write `Annotation` 
    objects to PyTables H5 files.
    '''

    def __init__(self, fpath, mode='a'):
        super(AnnotWriter, self).__init__(fpath, mode)
        self.table = None

    def create_table(self, table_name):
        '''Creates a new table with the given name.'''
        self.table = super(AnnotWriter, self)._create_table(table_name,
                                                            AnnotationDesc)

    def write(self, annotation):
        '''
        Writes an annotation to the *current* position of the
        table.

        Arguments
        ---------
        annotation: an `Annotation`
        '''
        self.table.row['DATE'] =  annotation.get_date()
        self.table.row['USER'] =  annotation.get_user()
        self.table.row['TAG'] =  annotation.get_tag()
        self.table.row['ITEM'] =  annotation.get_item()
        self.table.row.append()



class Annotation(BaseDao):
    '''
    Base object for an annotation
    '''

    def __init__(self, user, item, tag, date):
        super(Annotation, self).__init__()
        self.__user = user
        self.__item = item
        self.__tag = tag
        self.__date = date

        #Used for hashing and equality
        self.__tuple = (('DATE', date), ('ITEM', item),
                        ('TAG', tag), ('USER', user))

    def get_date(self):
        '''Return date in seconds since 1970'''
        return self.__date

    def get_user(self):
        '''Return user as int'''
        return self.__user

    def get_item(self):
        '''Return item as int'''
        return self.__item

    def get_tag(self):
        '''Return tag as int'''
        return self.__tag

    def get_tuple(self):
        '''Return tuple which represents annotation'''
        return self.__tuple

class AnnotationDesc(IsDescription):
    '''
    Defines an annotation description to be saved on file.
    '''

    DATE      = Time32Col() #@UndefinedVariable
    USER      = Int32Col() #@UndefinedVariable
    TAG       = Int32Col() #@UndefinedVariable
    ITEM      = Int32Col() #@UndefinedVariable