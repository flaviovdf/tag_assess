# -*- coding: utf8

'''
Module used for parsing annotation files and creating PyTables.
'''

from __future__ import print_function, division

import time

def delicious_flickr_parser(line):
    '''
    Parses a line from delicious and flickr annotations file
    
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
    tags = [x for x in spl[4:]]
    
    return (user, item, tags, date)

def parse(fpath):
    '''
    Parses input file
    
    Arguments:
    ----------
    fpath: str 
        Path to the annotation file
    '''
    
    
