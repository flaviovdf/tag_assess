# -*- coding: utf8
'''
Module used for parsing annotation files.
'''
from __future__ import print_function, division

from tagassess.common import ContiguousID

import time

def __convert_str_time(time_string, fmt):
    '''Converts the string to our time in local zone'''
    return time.mktime(time.strptime(time_string, fmt))

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
    date = __convert_str_time(spl[2].split(r'.')[0], '%Y-%m-%d %H:%M:%S')
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

    date = __convert_str_time('%s %s'%(mon_day_year, hour), '%Y-%m-%d %H:%M:%S')

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
    date = __convert_str_time(spl[2], '%Y-%m-%dT%H:%M:%SZ')
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

    date = __convert_str_time('%s %s'%(mon_day_year, hour), '%Y-%m-%d %H:%M:%S')

    user = int(spl[2])
    item = int(spl[3])
    tag = ' '.join(spl[4:])

    return (user, item, tag, date)

def to_json(user, item, tag, date):
    '''
    Creates a JSON dict for storing
    
    Arguments
    ---------
    user: any
        User name or id
    item: any
        Item name or id
    tag: any
        Tag name or id
    date: int (seconds)
        Annotation date
    '''
    
    return {'user':user,
            'item':item,
            'tag':tag,
            'date':date}

class Parser(object):

    '''
    Base class to parse annotation files. The main method of this class is `iparse`, which
    generates Annotations.
    '''

    def __init__(self):
        '''
        Creates a new Parser
        '''
        self.tag_ids = None
        self.item_ids = None
        self.user_ids = None
        self.__reset()

    def iparse(self, inf, parse_func, logf=None):
        '''
        Parses input file

        Arguments:
        ----------
        inf: file handle
            A text file handler which supports line iteration
        parse_func: callable
            Method or callable which will parse each line from the file
        '''
        for i, line in enumerate(inf):
            try:
                user, item, tag, date = parse_func(line)
                #We make use of tuple (1 to 3, X) in order to differentiate
                #user, tags and items with same names.
                yield to_json(self.user_ids[(1, user)],
                              self.item_ids[(2, item)],
                              self.tag_ids[(3, tag)], date)
            except Exception as exc:
                if logf:
                    print('Error at line %d:\n\t%s\nEx=%s'%(i, line, str(exc)))

    def __reset(self):
        '''Resets the parser ids to new ones. Useful for reusing
        the same parser object'''
        self.tag_ids = ContiguousID()
        self.item_ids = ContiguousID()
        self.user_ids = ContiguousID()