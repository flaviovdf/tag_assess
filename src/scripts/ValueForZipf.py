# -*- coding: utf8
'''
This script plot the value of each individual tags when considering a user
profile based on a zip-f distribution.
'''

from __future__ import division, print_function

from pymongo import Connection

from cy_tagassess import entropy
from tagassess.dao.mongodb.annotations import AnnotReader
from tagassess.index_creator import create_occurrence_index

import argparse
import numpy as np
import sys
import traceback

def main(annotations_dbname, annotations_tname, probabilities_dbname, 
         probabilities_tname, alpha):

    #Determine the items annotated by each tag
    tag_to_item = {}
    items = set()
    with AnnotReader(annotations_dbname) as reader:
        reader.change_table(annotations_tname)
        items.update(row['item'] for row in reader.iterate())
    
        #The index is created with sets, numpy has problems with this
        aux_index = create_occurrence_index(reader.iterate(), 'tag', 'item')
        for tag_id in aux_index:
            tag_to_item[tag_id] = np.array([item for item in aux_index[tag_id]])
    
    connection = None
    try:
        #Connects to DB
        connection = Connection()
        probabilties_database = connection[probabilities_dbname]
        if probabilities_tname not in probabilties_database.collection_names():
            print('Unable to find table in database', file=sys.stderr)
            return 2

        #Generates user profile based on zipf and computes value
        seeker_profile = np.array(np.random.zipf(alpha, len(items)), 'float64')
        seeker_profile /= seeker_profile.sum()
        
        probs_table = probabilties_database[probabilities_tname]
        for tag_id in tag_to_item:
            query = {'tag':tag_id}
            probs_iterator = probs_table.find(query)
            
            probs_it = np.array([row['prob_it'] for row in probs_iterator])
            dkl = entropy.kullback_leiber_divergence(probs_it, seeker_profile)
            rho = np.mean(seeker_profile[tag_to_item[tag_id]])
            
            print(tag_id, rho, dkl, rho*dkl)
    finally:
        if connection:
            connection.disconnect()

def create_parser(prog_name):
    desc = __doc__
    parser = argparse.ArgumentParser(prog_name, description=desc)
    parser.add_argument('annotations_dbname', type=str,
                        help='database to read annotations from')
    
    parser.add_argument('annotations_tname', type=str,
                        help='table with annotations')

    parser.add_argument('probabilities_dbname', type=str,
                        help='database to read probabilities from')

    parser.add_argument('probabilities_tname', type=str,
                        help='table with probabilities')

    parser.add_argument('alpha', type=float,
                        help='Parameter for the zip-f function')
    return parser

def entry_point(args=None):
    '''Fake main used to create argparse and call real one'''
    
    if not args: 
        args = []

    parser = create_parser(args[0])
    values = parser.parse_args(args[1:])
    
    try:
        return main(values.annotations_dbname, values.annotations_tname, 
                    values.probabilities_dbname, values.probabilities_tname, 
                    values.alpha)
    except:
        traceback.print_exc()
        parser.print_usage(file=sys.stderr)
        return 1

if __name__ == '__main__':
    sys.exit(entry_point(sys.argv))