# -*- coding: utf8
'''
Creates the tag tag graph.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

from tagassess import graph

import argparse
import traceback
import tempfile
import sys

def real_main(in_file, table, out_file, use):
    ntags, nsinks, iedges = \
     graph.iedge_list(in_file, table, use, uniq=False)
    n_nodes = ntags + nsinks
    
    tmp_fname = tempfile.mktemp()
    n_edges = 0
    with open(tmp_fname, 'w') as tmp:
        for source, dest in sorted(iedges):
            print(source, dest, file=tmp)
            n_edges += 1 
    
    with open(tmp_fname) as tmp:
        with open(out_file, 'w') as out:
            print('#Nodes:  %d'%n_nodes, file=out)
            print('#Edges:  %d'%n_edges, file=out)
            print('#Directed', file=out)
            for line in tmp:
                print(line[:-1], file=out)
                            
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Filters databases for exp.')
    
    parser.add_argument('in_file', type=str,
                        help='annotation h5 file to read from')

    parser.add_argument('table', type=str,
                        help='database table from the file')
    
    parser.add_argument('out_file', type=str,
                        help='new file for filtered')
    
    parser.add_argument('use', type=int, choices=[1, 2],
                        help='Use items = 1 or users = 2')
    
    return parser
    

def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.in_file, vals.table, vals.out_file,
                         vals.use)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))