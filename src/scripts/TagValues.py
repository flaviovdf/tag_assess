# -*- coding: utf8
'''
Simple scripts which prints the value of tags for a given user.
'''

from __future__ import division, print_function

from tagassess import value_calculator
from tagassess import smooth

import sys

def usage(prog_name, msg = None):
    '''Prints helps, msg if given and exits'''
    help_msg = 'Usage: %s <user> <smoothing type (jm | bayes)> <lambda> <num_relevant> <annotation_file> <table>'
    
    if msg:
        print(msg, file=sys.stderr)
    
    print(help_msg %prog_name, file=sys.stderr)
    return 1

def main(args=None):
    if not args: args = []
    
    if len(args) < 7:
        return usage(args[0])
        
    user = int(args[1])
    
    smoothing = args[2]
    if smoothing not in ('jm', 'bayes'):
        return usage(args[0], 'Unknown smoothing')
    
    smooth_func = None
    if smoothing == 'jm':
        smooth_func = smooth.jelinek_mercer
    else:
        smooth_func = smooth.bayes
    
    lambda_ = float(args[3])
    if (lambda_ <= 0 or lambda_ >= 1):
        return usage(args[0], 'Lambda should be in [0, 1]')
    
    num_relevant = int(args[4])
    annotation_file = args[5]
    table = args[6]
    
    itag_value = value_calculator.itag_value(annotation_file, table, 
                                             user, smooth_func, lambda_, 
                                             num_relevant)
    
    for tag_val, tag, used in sorted(itag_value, reverse=True):
        print(tag, tag_val, used)
                         
if __name__ == '__main__':
    sys.exit(main(sys.argv))