# -*- coding: utf8
from __future__ import division, print_function

__authors__ = ['Flavio Figueiredo - flaviovdf <at> gmail <dot-no-spam> com']
__date__ = '10/10/2011'

from tagassess import index_creator
from tagassess.dao.mongodb.annotations import AnnotReader

import argparse
import heapq
import itertools
import os
import sys
import traceback

def get_possible_queries(relevant_items_f):
    '''Reads the tags which lead to relevant items'''
    
    possible_queries = set()
    
    with open(relevant_items_f) as relevant_items:
        relevant_items.readline()
        for line in relevant_items:
            spl = line.split()
            tag = int(spl[1])
            
            possible_queries.add(tag)
    
    return possible_queries

def get_values_map(tag_values_f):
    '''Reads tag values'''
        
    pop_map = {}
    dkl_map = {}
    
    with open(tag_values_f) as tag_values:
        tag_values.readline()
        for line in tag_values:
            spl = line.split()
            
            tag = int(spl[0])
            pop = int(spl[1])
            dkl = float(spl[2])

            pop_map[tag] = pop
            dkl_map[tag] = dkl
    
    return pop_map, dkl_map

def get_user_relevant_items(info_f):
    '''Reads info file for user id and relevant items'''
    
    with open(info_f) as info:
        uid_line = info.readline()
        user = uid_line.split()[1]
        
        relevant_line = info.readline()
        relevant_items = set([int(iid) for iid in relevant_line.split()[4:]])

    return user, relevant_items

def build_cloud(tag_value_map, items, item_to_tag, max_size):
    '''Returns the set of tags in the cloud'''
    
    possible_tags = set()
    for item in items:
        for tag in item_to_tag[item]:
            possible_tags.add(tag)

    cloud = heapq.nlargest(max_size, possible_tags, 
                           key=lambda tag: tag_value_map[tag])
    
    return set(cloud)

def summarize_cloud(relevant_items, cloud, tag_to_item, tags_to_cover):
    '''Summarizes the cloud according to: precision, recall and coverage'''
    
    sum_precision = 0
    sum_recall = 0
    for tag in cloud:
        reachable = tag_to_item[tag]
        intersect = reachable.intersection(relevant_items)
        
        sum_precision += len(intersect) / len(reachable)
        sum_recall += len(intersect) / len(relevant_items)
        
    avg_precision = sum_precision / len(cloud)
    avg_recall = sum_recall / len(cloud)
    
    intersect_coverage = tags_to_cover.intersection(cloud)
    coverage = len(intersect_coverage) / len(tags_to_cover)
    
    return coverage, avg_precision, avg_recall

def real_main(database, table, user_folder):
    info_f = os.path.join(user_folder, 'info')
    relevant_items_f = os.path.join(user_folder, 'relevant_item.tags')
    tag_values_f = os.path.join(user_folder, 'tag.values')

    user, rel_items = get_user_relevant_items(info_f)
    possible_queries = get_possible_queries(relevant_items_f)
    pop_map, dkl_map = get_values_map(tag_values_f)
    
    with AnnotReader(database) as reader:
        reader.change_table(table)
        
        #Relevant items by user are left out with this query
        relevant_list = [item for item in rel_items]
        query = {'$or' : [
                          { 'user':{'$ne'  : user} }, 
                          { 'item':{'$nin' : relevant_list} }
                         ]
                }
        
        iterator = reader.iterate(query=query)
        tag_to_item, item_to_tag = \
            index_creator.create_double_occurrence_index(iterator, 
                                                          'tag', 
                                                          'item')

        
        print('#Query_size Query_Precision Query_Recall ' + \
              'Dkl_Size Dkl_Precision Dkl_Recall Dkl_Coverage' + \
              'Pop_Size Pop_Precision Pop_Recall Pop_Coverage')
        
        #Queries from size 1 to 3
        max_cloud_size = 20
        for query_size in xrange(1, 4):
            
            queries = itertools.combinations(possible_queries, query_size)
            for query in queries:
                
                #Simple and possibly inefficient AND search
                query_result = set()
                for term in query:
                    if len(query_result) == 0:
                        query_result.update(tag_to_item[term])
                    else:
                        query_result.intersection_update(tag_to_item[term])
                
                if len(query_result) == 0:
                    continue
                
                #Query results summary
                intersect = query_result.intersection(rel_items)
                precision = len(intersect) / len(query_result)
                recall = len(intersect) / len(rel_items)
                
                #Clouds summary
                dkl_cloud = build_cloud(dkl_map, query_result, item_to_tag, 
                                        max_cloud_size)
                pop_cloud = build_cloud(pop_map, query_result, item_to_tag, 
                                        max_cloud_size)
                
                dkl_coverage, dkl_precision, dkl_recall = \
                    summarize_cloud(rel_items, dkl_cloud, tag_to_item, 
                                    possible_queries)
                pop_coverage, pop_precision, pop_recall = \
                    summarize_cloud(rel_items, pop_cloud, tag_to_item, 
                                    possible_queries)
                
                print(len(query), precision, recall, 
                      len(dkl_cloud), dkl_precision, dkl_recall, dkl_coverage,
                      len(pop_cloud), pop_precision, pop_recall, pop_coverage)
                
                    
def create_parser(prog_name):
    desc = 'Computes coverage and precision for two clouds'
    parser = argparse.ArgumentParser(prog=prog_name,
                                     description=desc)
    
    parser.add_argument('database', type=str,
                        help='database to read from')
    
    parser.add_argument('table', type=str,
                        help='table with data')

    parser.add_argument('user_folder', type=str,
                        help='User folder')
    
    return parser
    

def main(args=None):
    if not args: args = []
    
    parser = create_parser(args[0])
    vals = parser.parse_args(args[1:])
    try:
        return real_main(vals.database, vals.table, 
                         vals.user_folder)
    except:
        parser.print_help()
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main(sys.argv))