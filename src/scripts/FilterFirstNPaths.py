#!/usr/bin/env python
'''
This script parses a totem output file filters
the first `N` shortest path for use in our baseline.
'''
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '26/05/2011'

import numpy as np

import argparse
import traceback
import sys

def real_main(in_file, out_file, num_consider):
    with open(in_file) as totem_file:
        data = {}
    
        #Initial line format =
        #Thu May 26 06:27:40 PDT 2011 - Running ./graph -b 82584 -e 199660
        line = totem_file.readline()
        spl = line.split()
        
        initial_item = int(spl[10])
        last_item = int(spl[12])

        num_tags = initial_item
        num_items = last_item - initial_item + 1
        
        while len(data) != num_consider:
            #There is no need to check EOF. Loop will always end.
            line = totem_file.readline()
            spl = line.split()
            
            tag = int(spl[0])
            item = int(spl[1])
            distance = int(spl[2])
            
            #Converts item id back!
            item_from_zero = item - num_tags
            if tag < num_tags:
                if tag not in data:
                    shape = (num_items, )
                    data[tag] = np.zeros(shape = shape)
                    
                assert data[tag][item_from_zero] == 0
                data[tag][item_from_zero] = distance
            else:
                #If we reach a tag greater than the number of tags
                #the rest of the file are items
                break
                
    with open(out_file, 'w') as new_out:
        for tag in data:
            paths = data[tag]
            
            #Reachable items.
            items_with_path = np.where(paths > 0)[0]
            
            #Has to reach at least 10 items.
            if len(items_with_path) > 10: 
                for item in items_with_path:
                    print(tag, item, paths[item], file=new_out)
                    
def create_parser(prog_name):
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description='Filters totem output.')
    
    parser.add_argument('in_file', type=str,
                        help='totem output file')

    parser.add_argument('out_file', type=str,
                        help='new file to create')

    parser.add_argument('num_consider', type=int,
                        help='Number of tags to consider')
        
    return parser
    
def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.in_file, vals.out_file, vals.num_consider)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))