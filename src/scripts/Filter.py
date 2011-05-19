# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict
from tagassess.dao.annotations import AnnotReader, AnnotWriter

import argparse
import traceback
import sys

def determine_good_users(in_file, table, min_pairs):
    pop_pairs = defaultdict(set)
    good_users = set()
    with AnnotReader(in_file) as reader:
        iterator = reader.iterate(table)
        for annotation in iterator:
            user = annotation.get_user()
            tag  = annotation.get_tag()
            item = annotation.get_item()
            
            if user not in good_users:
                pop_pairs[user].add((tag, item))
                if len(pop_pairs[user]) > min_pairs:
                    del pop_pairs[user]
                    good_users.add(user)
                    
    return good_users

def write_good_annots(in_file, table, out_file, min_users_per_item, 
                      good_users):
    pop_items = defaultdict(set)
    good_items = set()
    with AnnotReader(in_file) as reader, AnnotWriter(out_file) as writer:
        iterator = reader.iterate(table)
        writer.create_table(table)
        for annotation in iterator:
            user = annotation.get_user()
            item = annotation.get_item()
            
            if user in good_users and item not in good_items:
                pop_items[user].add(item)
                if len(pop_items[user]) > min_users_per_item:
                    del pop_items[user]
                    good_items.add(item)
                    writer.write(annotation)
            
def real_main(in_file, table, out_file, min_pairs, min_users_per_item):
    good_users = determine_good_users(in_file, table, min_pairs)
    write_good_annots(in_file, table, out_file, min_users_per_item, good_users)
            
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Computes the baseline.')
    
    parser.add_argument('in_file', type=str,
                        help='annotation h5 file to read from')

    parser.add_argument('table', type=str,
                        help='database table from the file')
    
    parser.add_argument('out_file', type=str,
                        help='new file for filtered')
    
    parser.add_argument('min_pairs', type=int,
                        help='Min unique tag x item pairs per user')

    parser.add_argument('min_users_per_item', type=int,
                        help='Min users who tagged an item')
    
    return parser
    

def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.in_file, vals.table, vals.out_file,
                         vals.min_pairs, vals.min_users_per_item)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))