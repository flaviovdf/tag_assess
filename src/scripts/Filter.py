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
from tagassess.common import ContiguousID
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.dao.mongodb.annotations import AnnotWriter

import argparse
import traceback
import sys
import os

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

    return good_items

def write_good_annots(database, table, new_database, good_items):
    '''Writes new annotations based on filters'''
    user_ids = ContiguousID()
    tag_ids = ContiguousID()
    item_ids = ContiguousID()

    with AnnotReader(database) as reader, AnnotWriter(new_database) as writer:
        reader.change_table(table)
        writer.create_table(table)
        iterator = reader.iterate()
        for annotation in iterator:
            user = annotation['user']
            item = annotation['item']
            tag  = annotation['tag']
            date = annotation['date']
            
            if item in good_items:
                new_annot = data_parser.to_json(user_ids[(1, user)],
                                                item_ids[(2, item)],
                                                tag_ids[(3, tag)], date)
                writer.append_row(new_annot)

    return user_ids, item_ids, tag_ids

def real_main(in_file, table, out_file, min_users_per_item,
              new_ids_folder):
    '''Main'''
    good_items = determine_good_items(in_file, table,
                                      min_users_per_item)
    
    user_ids, item_ids, tag_ids = \
        write_good_annots(in_file, table, out_file, good_items)
    
    with open(os.path.join(new_ids_folder, table + '.user'), 'w') as userf:
        print('old_id', 'new_id', file=userf)
        for user in sorted(user_ids, key=user_ids.__getitem__):
            print(user[1], user_ids[user], file=userf)

    with open(os.path.join(new_ids_folder, table + '.items'), 'w') as itemsf:
        print('old_id', 'new_id', file=itemsf)
        for item in sorted(item_ids, key=item_ids.__getitem__):
            print(item[1], item_ids[item], file=itemsf)

    with open(os.path.join(new_ids_folder, table + '.tags'), 'w') as tagsf:
        print('old_id', 'new_id', file=tagsf)
        for tag in sorted(tag_ids, key=tag_ids.__getitem__):
            print(tag[1], tag_ids[tag], file=tagsf)
            
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
    
    parser.add_argument('new_ids_folder', type=str,
                        help='Folder for new id mappings')
    
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
                         vals.min_users_per_item, vals.new_ids_folder)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))