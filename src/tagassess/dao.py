# -*- coding: utf8
'''
Contains base classes for accessing annotation stored on PyTables files.
'''

from __future__ import print_function, division

from tables import *

class Annotation(object):
    '''
    Base object for an annotation
    '''
    
    def __init__(self, user, item, tag, date):
        self._user = user
        self._item = item
        self._tag = tag
        self._date = date
    
    def get_date(self):
        return self._date
    
    def get_time(self):
        return self._date
    
    def get_user(self):
        return self._user
    
    def get_item(self):
        return self._item
    
    def get_tag(self):
        return self._tag
    
    def get_desc(self):
        return {'DATE':self.get_date(),
                'USER':self.get_user(),
                'TAG':self.get_tag(),
                'ITEM':self.get_time()
                }

class AnnotationDesc(IsDescription):
    
    '''
    Defines an annotation description to be saved on file.
    '''
    
    DATE      = Int32Col() #@UndefinedVariable
    USER      = Int32Col() #@UndefinedVariable
    TAG       = Int32Col() #@UndefinedVariable
    ITEM      = Int32Col() #@UndefinedVariable