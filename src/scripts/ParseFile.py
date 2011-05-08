# -*- coding: utf8
'''
ParseFile
=========

Script which parsers annotation files
'''

from __future__ import print_function, division

from tagassess import data_parser
from tagassess.dao.annotations import AnnotWriter

import os
import sys

def main(args=[]):

    if len(args) < 5:
        print('Usage %s %s %s %s %s'
              %(args[0], '<annotation_file>', '<output_h5file>', '<ids_folder>',
                '<ftype = {flickr, delicious, bibsonomy, connotea, citeulike}'),
                file=sys.stderr)
        return 1

    func_map = {'bibsonomy':data_parser.bibsonomy_parser,
                'citeulike':data_parser.citeulike_parser,
                'connotea':data_parser.connotea_parser,
                'delicious':data_parser.delicious_flickr_parser,
                'flickr':data_parser.delicious_flickr_parser}

    infpath = args[1]
    outh5f = args[2]
    idsfold = args[3]

    func_name = args[4]
    if func_name not in func_map:
        print('ftype %s unknown'%func_name)
        return 1
    parse_func = func_map[func_name]

    table_file = None
    try:
        #Saving Table to H5 file
        parser = data_parser.Parser()
        with open(infpath) as annotf, AnnotWriter(outh5f, 'a') as writer:
            writer.create_table(func_name)
            for annotation in parser.iparse(annotf, parse_func):
                writer.write(annotation)

        #Saving IDs to text files
        user_ids = parser.user_ids
        item_ids = parser.item_ids
        tag_ids = parser.tag_ids

        with open(os.path.join(idsfold, func_name + '.user'), 'w') as userf:
            for user in sorted(user_ids, key=user_ids.__getitem__):
                print(user[1], user_ids[user], file=userf)

        with open(os.path.join(idsfold, func_name + '.items'), 'w') as itemsf:
            for item in sorted(item_ids, key=item_ids.__getitem__):
                print(item[1], item_ids[item], file=itemsf)

        with open(os.path.join(idsfold, func_name + '.tags'), 'w') as tagsf:
            for tag in sorted(tag_ids, key=tag_ids.__getitem__):
                print(tag[1], tag_ids[tag], file=tagsf)
    finally:
        if table_file: table_file.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
