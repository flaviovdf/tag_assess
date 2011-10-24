# -*- coding: utf8
'''
This script dumps a Mongo database table to a CSV format.
'''
from __future__ import division, print_function

from tagassess.dao.mongodb.annotations import AnnotReader

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '23/10/2011'

import argparse
import sys
import traceback

def main(database, table):
    with AnnotReader(database) as reader:
        reader.change_table(table)
        print('tag', 'item', 'user', 'date', sep=',')
        for row in table:
            print(row['tag'], row['item'], row['user'], row['date'], sep=',')

def create_parser(prog_name):
    desc = __doc__
    parser = argparse.ArgumentParser(prog_name, description=desc)
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.database, values.table)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))