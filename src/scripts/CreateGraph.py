# -*- coding: utf8
'''
Creates the tag tag graph.
'''

from __future__ import division, print_function

from tagassess import graph

import sys

def usage(prog_name, msg = None):
    '''Prints helps, msg if given and exits'''
    help_msg = 'Usage: %s <annotation_file> <table> <use: items or users>'
    
    if msg:
        print(msg, file=sys.stderr)
    
    print(help_msg %prog_name, file=sys.stderr)
    return 1

def main(args=None):
    if not args: args = []
    
    if len(args) < 4:
        return usage(args[0])
        
    annotation_file = args[1]
    table = args[2]
    collapse_on = args[3]
    
    use = 0
    if collapse_on == 'items':
        use = 1
    elif collapse_on == 'users':
        use = 2
    else:
        usage(args[0], 'Unknown option %s'%collapse_on)
        return 1
    
    base_index = \
     graph.extract_indexes_from_file(annotation_file, table, use)

    tag_nodes, sink_nodes, edge_list = \
     graph.edge_list(base_index, uniq=False)
    n_nodes = len(tag_nodes) + len(sink_nodes)
    
    print('#Nodes:  %d'%n_nodes)
    print('#Edges:  %d'%len(edge_list))
    print('#Directed')
    for source, dest in edge_list:
        print(source, dest) 
                            
if __name__ == '__main__':
    sys.exit(main(sys.argv))