# -*- coding: utf8
#pylint: disable-msg=W0401
#pylint: disable-msg=W0614

'''
Contains base classes for accessing annotation stored on PyTables files.
'''

from __future__ import print_function, division

from collections import Iterable
#Wildcard necessary for PyTables.
from tables import * 

import itertools

class FileNotOpenException(Exception):
    '''Used to notify of file not open errors'''
    pass

class AnnotReader(Iterable):
    '''A AnnotReader is used to read PyTables H5 files.'''

    def __init__(self, fpath, table_name):
        self.fpath = fpath
        self.table_name = table_name
        self.opened = False
        self.table = None
        self.tablefile = None

    def __chk_opened(self):
        '''
        Checks if the file is opened and raises an exception if not.
        '''
        if not self.opened:
            raise FileNotOpenException(
                    'File %s has not been open. Use AnnotWriter.openfile()')

    def openfile(self):
        '''Opens the file given in the constructor.'''
        self.tablefile = openFile(self.fpath, 'r')
        self.table = self.tablefile.getNode('/', self.table_name)
        self.opened = True

    def closefile(self):
        '''
        Closes the file.

        Raises
        ------
            `FileNotOpenException` if the file has not been opened
        '''
        self.__chk_opened()
        self.tablefile.close()
        self.opened = False
        self.table = None

    def __enter__(self):
        self.openfile()
        return self

    def __exit__(self, type, value, traceback):
        '''Closes the file return `True` if no exeception was caught'''
        self.closefile()
        return not value #Value is an exception in case with fails.

    def __iter__(self):
        '''Creates a iterator over the table'''
        return itertools.imap(lambda row:
                                Annotation(row['USER'], row['ITEM'],
                                           row['TAG'], row['DATE']),
                              self.table)

class AnnotWriter(object):

    '''
    A AnnotWriter is used to create PyTables H5 files. Basically,
    it is used to write new annotations to the file.
    '''

    def __init__(self, fpath, mode='a'):
        self.fpath = fpath
        self.mode = mode
        self.opened = False
        self.tablefile = None
        self.table = None

    def __chk_opened(self):
        '''
        Checks if the file is opened and raises an exception if not.
        '''
        if not self.opened:
            raise FileNotOpenException(
                    'File %s has not been open. Use AnnotWriter.openfile()')

    def create_table(self, tname):
        '''
        Creates a new table in the annotation H5 file.

        Arguments
        ---------
        tname: str
            The name of the new table

        Raises
        ------
            `FileNotOpenException` if the file has not been opened
        '''
        self.__chk_opened()
        self.table = self.tablefile.createTable(self.tablefile.root,
                                                tname, AnnotationDesc)

    def openfile(self):
        '''Opens the file given in the constructor.'''
        self.tablefile = openFile(self.fpath, self.mode)
        self.opened = True

    def closefile(self):
        '''
        Closes the file.

        Raises
        ------
            `FileNotOpenException` if the file has not been opened
        '''
        self.__chk_opened()
        self.tablefile.close()
        self.opened = False
        self.table = None

    def write(self, annotation):
        '''
        Writes an annotation to the *current* position of the
        table file.

        Arguments
        ---------
        annotation: an `Annotation`
        '''
        self.__chk_opened()

        self.table.row['DATE'] =  annotation.get_date()
        self.table.row['USER'] =  annotation.get_user()
        self.table.row['TAG'] =  annotation.get_tag()
        self.table.row['ITEM'] =  annotation.get_item()
        self.table.row.append()

    def __enter__(self):
        self.openfile()
        return self

    def __exit__(self, type, value, traceback):
        '''Closes the file return `True` if no exeception was caught'''
        self.closefile()
        return not value #Value is an exception in case with fails.

class Annotation(object):
    '''
    Base object for an annotation
    '''

    def __init__(self, user, item, tag, date):
        self.__user = user
        self.__item = item
        self.__tag = tag
        self.__date = date

        #Used for hashing and equality
        self.__tuple = (('DATE', date), ('ITEM', item),
                        ('TAG', tag), ('USER', user))

    def get_date(self):
        '''Return date in seconds'''
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

    def __str__(self):
        return str(self.get_tuple())

    def __eq__(self, other):
        if isinstance(other, Annotation):
            return self.get_tuple() == other.get_tuple()
        else:
            return False

    def __hash__(self):
        return hash(self.get_tuple())


class AnnotationDesc(IsDescription):

    '''
    Defines an annotation description to be saved on file.
    '''

    DATE      = Time32Col() #@UndefinedVariable
    USER      = Int32Col() #@UndefinedVariable
    TAG       = Int32Col() #@UndefinedVariable
    ITEM      = Int32Col() #@UndefinedVariable