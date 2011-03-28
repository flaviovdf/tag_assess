# -*- coding: utf8
'''
ParseFile
=========

Script which parsers annotation files
'''

from __future__ import print_function, division

from tagassess import data_parser
from tagassess.dao import AnnotationDesc

import sys
import tables

def main(args=[]):
    
    if len(args) < 4:
        print('Usage %s %s %s %s'
              %(args[0], '<annotation_file>', '<output_h5file>', 
                '<ftype = {flickr, delicious, bibsonomy, connotea, citeulike}'), 
                sys.stderr)
        return 1
    
    func_map = {'bibsonomy':data_parser.bibsonomy_parser,
                'citeulike':data_parser.citeulike_parser,
                'connotea':data_parser.connotea_parser,
                'delicious':data_parser.delicious_flickr_parser,
                'flickr':data_parser.delicious_flickr_parser}
    
    infpath = args[1]
    
    func_name = args[2]
    if args[2] not in func_map:
        print('ftype %s unknown'%func_name)
        return 1
    parse_func = func_map[args[2]]
    
    outf = args[3]
    
    table_file = None
    try:
        table_file = tables.openFile(outf, 'a')
        root = table_file.root
        table = table_file.createTable(root, func_name, AnnotationDesc)
        parser = data_parser.Parser()
        with open(infpath) as annotf:
            for annotation in parser.iparse(annotf, parse_func):
                row = table.row
                row.append(annotation.get_desc())
    finally:
        if table_file: table_file.close()
        
if __name__ == '__main__':
    sys.exit(main(sys.argv))