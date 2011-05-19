# -*- coding: utf8

from __future__ import division, print_function

from collections import defaultdict
from tagassess.dao.annotations import AnnotReader, AnnotWriter

import sys

def usage(prog_name, msg = None):
    '''Prints helps, msg if given and exits'''
    help_msg = 'Usage: %s <in_file> <in-table> <out_file> <user | tag | item> <count>'
    
    if msg:
        print(msg, file=sys.stderr)
    
    print(help_msg %prog_name, file=sys.stderr)
    return 1

def main(args=None):
    if not args: args = []
    
    if len(args) < 6:
        return usage(args[0])
        
    in_file = args[1]
    table = args[2]
    out_file = args[3]
    filter_col = args[4]
    count = int(args[5])
    
    filter_funcs = {'tag':lambda annot: annot.get_tag(),
                    'item':lambda annot: annot.get_item(),
                    'user':lambda annot: annot.get_user()}
    filter_func = filter_funcs[filter_col]
    
    pop = defaultdict(int)
    with AnnotReader(in_file) as reader, AnnotWriter(out_file) as writer:
        iterator = reader.iterate(table)
        writer.create_table(table)
        for annotation in iterator:
            key = filter_func(annotation)
            pop[key] += 1
            
            if pop[key] > count:
                writer.write(annotation)
                    
if __name__ == '__main__':
    sys.exit(main(sys.argv))