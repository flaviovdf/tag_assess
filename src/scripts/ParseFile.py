# -*- coding: utf8
'''
ParseFile
=========

Script which parsers annotation files
'''
from __future__ import print_function, division

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from tagassess import data_parser
from tagassess.dao.pytables.annotations import AnnotWriter

import os
import sys

def main(args=[]):

    if len(args) < 4:
        types = '{flickr, delicious, bibsonomy, connotea, citeulike, lt}'
        print('Usage %s %s %s %s %s'
              %(args[0], '<annotation_file>', '<database_file>', '<ids folder>',
                '<ftype = %s>' % types),
                file=sys.stderr)
        return 1

    func_map = {'bibsonomy':data_parser.bibsonomy_parser,
                'citeulike':data_parser.citeulike_parser,
                'connotea':data_parser.connotea_parser,
                'delicious':data_parser.delicious_flickr_parser,
                'flickr':data_parser.delicious_flickr_parser,
                'lt':data_parser.library_thing_parser}
    
    in_fpath = args[1]
    db_fpath = args[2]
    ids_folder = args[3]
    func_name = args[4]
    
    if func_name not in func_map:
        print('ftype %s unknown'%func_name)
        return 1
    parse_func = func_map[func_name]

    #Saving Table to PyTables
    parser = data_parser.Parser()
    with open(in_fpath) as annotf, AnnotWriter(db_fpath) as writer:
        writer.create_table(func_name)
        
        for annotation in parser.iparse(annotf, parse_func, sys.stderr):
            writer.append_row(annotation)

    #Saving IDs to text files
    user_ids = parser.user_ids
    item_ids = parser.item_ids
    tag_ids = parser.tag_ids

    with open(os.path.join(ids_folder, func_name + '.user'), 'w') as userf:
        for user in sorted(user_ids, key=user_ids.__getitem__):
            print(user[1], user_ids[user], file=userf)

    with open(os.path.join(ids_folder, func_name + '.items'), 'w') as itemsf:
        for item in sorted(item_ids, key=item_ids.__getitem__):
            print(item[1], item_ids[item], file=itemsf)

    with open(os.path.join(ids_folder, func_name + '.tags'), 'w') as tagsf:
        for tag in sorted(tag_ids, key=tag_ids.__getitem__):
            print(tag[1], tag_ids[tag], file=tagsf)

if __name__ == '__main__':
    sys.exit(main(sys.argv))