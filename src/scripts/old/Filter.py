# -*- coding: utf8
'''
This Scripts filters users and items from the trace based the amount
of users who tagged an item.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from collections import defaultdict

from tagassess import data_parser
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.dao.mongodb.annotations import AnnotWriter
from tagassess.dao.mongodb.keyval import KeyValStore

import argparse
import traceback
import sys

def determine_good_items(database, table, min_users_per_item):
    '''Filters the items used by a minimum number of users'''
    pop_items = defaultdict(set)
    good_items = set()
    with AnnotReader(database) as reader:
        reader.change_table(table)
        iterator = reader.iterate()
        for annotation in iterator:
            user = annotation['user']
            item = annotation['item']
            
            if item not in good_items:
                pop_items[item].add(user)
                if len(pop_items[item]) >= min_users_per_item:
                    del pop_items[item]
                    good_items.add(item)
                    
    return [item for item in sorted(good_items)]

def write_good_annots(database, table, new_database, good_items):
    '''Writes new annotations based on filters'''
    with AnnotReader(database) as reader, AnnotWriter(new_database) as writer:
        reader.change_table(table)
        writer.create_table(table)
        iterator = reader.iterate(query = {'item': {'$in' : good_items } })
        
        parser = data_parser.Parser()
        iparse = parser.iparse(iterator, data_parser.json_parser)
        for new_annot in iparse:
            writer.append_row(new_annot)

    return parser.user_ids, parser.item_ids, parser.tag_ids

def real_main(database, table, new_database, min_users_per_item):
    '''Main'''
    good_items = determine_good_items(database, table,
                                      min_users_per_item)
    
    user_ids, item_ids, tag_ids = \
        write_good_annots(database, table, new_database, good_items)
    
    with KeyValStore(new_database) as new_ids, KeyValStore(database) as old_ids:
        new_ids.create_table(table + '_user_ids')
        old_ids.change_table(table + '_user_ids')
        cache = old_ids.get_all()
        for old_uid, new_uid in user_ids.items():
            new_ids.put(new_uid, cache[old_uid[1]], no_check = True)

        new_ids.create_table(table + '_item_ids')
        old_ids.change_table(table + '_item_ids')
        cache = old_ids.get_all()
        for old_iid, new_iid in item_ids.items():
            new_ids.put(new_iid, cache[old_iid[1]], no_check = True)

        new_ids.create_table(table + '_tag_ids')
        old_ids.change_table(table + '_tag_ids')
        cache = old_ids.get_all()
        for old_tid, new_tid in tag_ids.items():
            new_ids.put(new_tid, cache[old_tid[1]], no_check = True)
            
def create_parser(prog_name):
    '''Adds command line options'''
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Filters databases for exp.')
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')
    
    parser.add_argument('new_database', type=str,
                        help='database to store filtered annotations')
    
    parser.add_argument('min_users_per_item', type=int,
                        help='Min users who tagged an item')
    
    return parser
    
def main(args=None):
    '''Fake Main'''
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.database, vals.table, vals.new_database,
                         vals.min_users_per_item)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))