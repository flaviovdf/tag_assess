# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict

from tagassess.common import ContiguousID
from tagassess.dao.annotations import AnnotReader, AnnotWriter, Annotation

import argparse
import traceback
import sys
import os

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
                if len(pop_pairs[user]) >= min_pairs:
                    del pop_pairs[user]
                    good_users.add(user)
                    
    return good_users

def write_good_annots(in_file, table, out_file, min_users_per_item, 
                      good_users):
    
    user_ids = ContiguousID()
    tag_ids = ContiguousID()
    item_ids = ContiguousID()
    
    pop_items = defaultdict(set)
    good_items = set()
    with AnnotReader(in_file) as reader, AnnotWriter(out_file) as writer:
        iterator = reader.iterate(table)
        writer.create_table(table)
        for annotation in iterator:
            user = annotation.get_user()
            item = annotation.get_item()
            tag  = annotation.get_item()
            date  = annotation.get_date()
            
            if user in good_users:
                if item in good_items:
                    writer.write(annotation)
                else:
                    pop_items[user].add(item)
                    if len(pop_items[user]) >= min_users_per_item:
                        del pop_items[user]
                        good_items.add(item)
                        
                        new_annot = Annotation(user_ids[(1, user)],
                                               item_ids[(2, item)],
                                               tag_ids[(3, tag)], date)
                        writer.write(new_annot)
    
    return user_ids, item_ids, tag_ids

def old_ids_to_dict(id_path):
    rv = {}
    with open(id_path) as f:
        for l in f:
            spl = l.split()
            k = ' '.join(str(x) for x in spl[:-1])
            v = int(spl[-1])
            rv[v] = k
    return rv
            
def real_main(in_file, table, out_file, min_pairs, min_users_per_item,
              old_ids_folder, new_ids_folder):
    
    good_users = determine_good_users(in_file, table, min_pairs)
    user_ids, item_ids, tag_ids = \
        write_good_annots(in_file, table, out_file, min_users_per_item, good_users)
    
    old_user_ids = old_ids_to_dict(os.path.join(old_ids_folder, table + '.user'))
    old_item_ids = old_ids_to_dict(os.path.join(old_ids_folder, table + '.items'))
    old_tag_ids = old_ids_to_dict(os.path.join(old_ids_folder, table + '.tags'))
    
    with open(os.path.join(new_ids_folder, table + '.user'), 'w') as userf:
        for user in sorted(user_ids):
            print(old_user_ids[user_ids[user]], user_ids[user], file=userf)

    with open(os.path.join(new_ids_folder, table + '.items'), 'w') as itemsf:
        for item in sorted(item_ids):
            print(old_item_ids[item_ids[item]], item_ids[item], file=itemsf)

    with open(os.path.join(new_ids_folder, table + '.tags'), 'w') as tagsf:
        for tag in sorted(tag_ids):
            print(old_tag_ids[tag_ids[tag]], tag_ids[tag], file=tagsf)
            
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Computes the baseline.')
    
    parser.add_argument('in_file', type=str,
                        help='annotation h5 file to read from')

    parser.add_argument('table', type=str,
                        help='database table from the file')
    
    parser.add_argument('out_file', type=str,
                        help='new file for filtered')
    
    parser.add_argument('old_ids_folder', type=str,
                        help='Folder with old id mappings')

    parser.add_argument('new_ids_folder', type=str,
                        help='Folder for new id mappings')
    
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
                         vals.min_pairs, vals.min_users_per_item,
                         vals.old_ids_folder, vals.new_ids_folder)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))