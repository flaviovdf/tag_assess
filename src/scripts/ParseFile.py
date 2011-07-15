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
from tagassess.dao.mongodb.annotations import AnnotWriter
from tagassess.dao.mongodb.keyval import KeyValStore

import sys

def main(args=[]):

    if len(args) < 4:
        print('Usage %s %s %s %s'
              %(args[0], '<annotation_file>', '<database_name>',
                '<ftype = {flickr, delicious, bibsonomy, connotea, citeulike}'),
                file=sys.stderr)
        return 1

    func_map = {'bibsonomy':data_parser.bibsonomy_parser,
                'citeulike':data_parser.citeulike_parser,
                'connotea':data_parser.connotea_parser,
                'delicious':data_parser.delicious_flickr_parser,
                'flickr':data_parser.delicious_flickr_parser}
    
    infpath = args[1]
    database_name = args[2]

    func_name = args[3]
    if func_name not in func_map:
        print('ftype %s unknown'%func_name)
        return 1
    parse_func = func_map[func_name]

    table_file = None
    try:
        #Saving Table to MongoDB
        parser = data_parser.Parser()
        with open(infpath) as annotf, AnnotWriter(database_name) as writer:
            writer.create_table(func_name)
            
            for annotation in parser.iparse(annotf, parse_func, sys.stderr):
                writer.append_row(annotation)

        #Saving IDs to text files
        user_ids = parser.user_ids
        item_ids = parser.item_ids
        tag_ids = parser.tag_ids
    
        with KeyValStore(database_name) as keyval:
            keyval.create_table(func_name + '_user_ids')
            for user, uid in user_ids.items():
                keyval.put(uid, user[1].strip(), no_check = True)
    
            keyval.create_table(func_name + '_item_ids')
            for item, iid in item_ids.items():
                keyval.put(iid, item[1].strip(), no_check = True)
    
            keyval.create_table(func_name + '_tag_ids')
            for tag, tid in tag_ids.items():
                keyval.put(tid, tag[1].strip(), no_check = True)
    finally:
        if table_file: 
            table_file.close()

if __name__ == '__main__':
    sys.exit(main(sys.argv))
