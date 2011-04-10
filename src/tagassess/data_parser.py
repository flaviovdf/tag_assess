# -*- coding: utf8
'''
Module used for parsing annotation files and creating PyTables.
'''

from __future__ import print_function, division

from .common import ContiguousID
from .dao import Annotation

import time

def citeulike_parser(line):
    '''
    Parses a line from CiteULike annotations file
    
    Arguments
    ---------
    line: str
        the line to parse
    '''
    spl = line.split('|')
    
    item = int(spl[0])
    user = spl[1]
    date = time.strptime(spl[2].split(r'.')[0], '%Y-%m-%d %H:%M:%S')
    tag = spl[3]
    
    return (user, item, tag, date)
    
def bibsonomy_parser(line):
    '''
    Parses a line from Bibsonomy annotations file
    
    Arguments
    ---------
    line: str
        the line to parse
    '''
    spl = line.split()
    
    user = int(spl[0])
    tag = spl[1]
    item = int(spl[2])
     
    mon_day_year = spl[4]
    hour = spl[5]
    
    date = time.strptime('%s %s'%(mon_day_year, hour), '%Y-%m-%d %H:%M:%S')
    
    return (user, item, tag, date)

def connotea_parser(line):
    '''
    Parses a line from Connotea annotations file
    
    Arguments
    ---------
    line: str
        the line to parse
    '''
    spl = line.split('|')
    
    item = spl[0]
    user = spl[1]
    date = time.strptime(spl[2], '%Y-%m-%dT%H:%M:%SZ')
    tag = spl[3]
    
    return (user, item, tag, date)

def delicious_flickr_parser(line):
    '''
    Parses a line from Delicious and Flickr annotations file
    
    Arguments
    ---------
    line: str
        the line to parse
    '''
    spl = line.split()
    
    mon_day_year = spl[0]
    hour = spl[1]
    
    date = time.strptime('%s %s'%(mon_day_year, hour), '%Y-%m-%d %H:%M:%S')
    
    user = int(spl[2])
    item = int(spl[3])
    tag = ' '.join(spl[4:])
    
    return (user, item, tag, date)

class Parser(object):
    
    '''
    Base class to parse annotation files. The main method of this class is `iparse`, which
    generates Annotations.
    '''
    
    def __init__(self, share_ids=False):
        '''
        Creates a new Parser
        
        Arguments:
        ----------
        share_ids: bool
            Determines if tags, items and user will share the same id space
        '''
        self.share_ids = share_ids
        self.tag_ids = None
        self.item_ids = None
        self.user_ids = None
        self._reset()
        
    def iparse(self, inf, parse_func):
        '''
        Parses input file
        
        Arguments:
        ----------
        inf: file handle 
            A text file handler which supports line iteration
        parse_func: callable
            Method or callable which will parse each line from the file
        '''
        for line in inf:
            user, item, tag, date = parse_func(line)
            #We make use of tuple (1 to 3, X) in order to differentiate
            #user, tags and items with same names. 
            yield Annotation(self.user_ids[(1, user)], 
                             self.item_ids[(2, item)], 
                             self.tag_ids[(3, tag)], date)
    
    def _reset(self):
        if not self.share_ids:
            self.tag_ids = ContiguousID()
            self.item_ids = ContiguousID()
            self.user_ids = ContiguousID()
        else:
            ids = ContiguousID()
            self.tag_ids = ids
            self.item_ids = ids
            self.user_ids = ids