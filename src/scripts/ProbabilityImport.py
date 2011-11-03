# -*- coding: utf8
'''
This script imports pre computed probabilities to a Mongo Database
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '03/11/2011'

from pymongo import ASCENDING
from pymongo import Connection

import argparse
import sys
import traceback

TAG_ID = 0
ITEM_ID = 1
PROB = 3

def main(precomputed_fpath, db_name, tname):
    with open(precomputed_fpath) as precomputed_file:
        connection = None
        database = None
        try:
            connection = Connection()
            database = connection[db_name]
            
            if tname in database.collection_names():
                print('Table already exists', file=sys.stderr)
                return 2
            
            table = database[tname]
            table.ensure_index([('tag', ASCENDING), ('item', ASCENDING)])
            
            for line in precomputed_file:
                if '#' in line:
                    continue
                
                spl = line.split('|')
                
                tag_id = int(spl[TAG_ID])
                item_id = int(spl[ITEM_ID])
                prob = float(spl[PROB])
                
                table.insert({'tag':tag_id, 'item':item_id, 'pit':prob})
                
        finally:
            if connection:
                connection.disconnect()

def create_parser(prog_name):
    desc = __doc__
    parser = argparse.ArgumentParser(prog_name, description=desc)
    parser.add_argument('precomputed_fpath', type=str,
                        help='the file to read from')
    
    parser.add_argument('db_name', type=str,
                        help='database to use')
    
    parser.add_argument('tname', type=str,
                        help='table to create with probabilities')
    
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.precomputed_fpath, values.db_name, values.tname)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))