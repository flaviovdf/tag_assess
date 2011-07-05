# -*- coding: utf8
#pylint: disable-msg=W0614
'''Classes for reading annotations from PyTables'''
from __future__ import division, print_function

from tagassess.dao.base import Reader
from tagassess.dao.base import Writer
from tables import *

import itertools

class AnnotReader(Reader):
    '''
    A `AnnotReader` is used to read PyTables H5 files.
    '''
    def __init__(self, fpath):
        super(AnnotReader, self).__init__()
        self.fpath = fpath
        self.opened = False
        self.tablefile = None
        self.table = None
        
    def open(self):
        if not self.opened:
            self.tablefile = openFile(self.fpath, 'r')
            self.opened = True
            self.table = None

    def close(self):
        if self.opened:
            self.tablefile.close()
            self.opened = False
            self.table = None

    def change_table(self, tname, **kwargs):
        self.table = self.tablefile.getNode('/', tname)
        
    def iterate(self, query = None, **kwargs):
        iterable = None
        if query:
            iterable = self.table.where(query)
        else:
            iterable = self.table
        
        conv = lambda row: {'user':row['user'],
                            'item':row['item'],
                            'tag': row['tag'],
                            'date':row['date']}
        
        return itertools.imap(conv, iterable)
    
class AnnotWriter(Writer):
    '''
    A `AnnotWriter` is used to create and write `Annotation` 
    objects to PyTables H5 files.
    '''

    def __init__(self, fpath, mode='a'):
        super(AnnotWriter, self).__init__()
        self.fpath = fpath
        self.mode = mode
        self.opened = False
        self.tablefile = None
        self.table = None
        
    def open(self):
        if not self.opened:
            self.tablefile = openFile(self.fpath, self.mode)
            self.opened = True
            self.table = None

    def close(self):
        if self.opened:
            self.tablefile.close()
            self.opened = False
            self.table = None

    def change_table(self, tname, **kwargs):
        self.table = self.tablefile.getNode('/', tname)

    def create_table(self, tname, **kwargs):
        self.table = self.tablefile.createTable(self.tablefile.root,
                                                tname, AnnotationDesc)

    def append_row(self, row, **kwargs):
        self.table.row['date'] = row['date']
        self.table.row['user'] = row['user']
        self.table.row['tag']  = row['tag']
        self.table.row['item'] = row['item']
        self.table.row.append()

class AnnotationDesc(IsDescription):
    '''
    Defines an annotation description to be saved on file.
    '''
    date      = Time32Col() #@UndefinedVariable
    user      = Int32Col() #@UndefinedVariable
    tag       = Int32Col() #@UndefinedVariable
    item      = Int32Col() #@UndefinedVariable