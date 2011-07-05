# -*- coding: utf8
'''Functions for creating tag graphs based on co-occurrence in items or users'''

from __future__ import division, print_function

from tagassess.index_creator import create_double_occurrence_index

import networkx as nx

def iedge_from_annotations(annotation_it, use=1):
    '''
    Returns the edge list for the navigational graph.
    
    Arguments
    ---------
    annotation_it: iterator
        Iterator to annotations to use
    use = int {1, 2}
        Indicates whether to use items or users:
            1: Items
            2: Users
    uniq: bool
        Indicates if ids in indices are already unique, that is no 
        tag, sink and user shares the same id.
    '''
    choices = {1:'item',
               2:'user'}
    dest = choices[use]
    
    tag_index, sink_index = create_double_occurrence_index(annotation_it, 
                                                           'tag', dest)
    return iedge_from_indexes(tag_index, sink_index)

def iedge_from_indexes(tag_index, sink_index):
    '''
    Returns the edge list for the navigational graph.
    
    Arguments
    ---------
    tag_index: dict (int to set<int>)
        Tag node to sink nodes index
    sink_index = dict (int to set<int>)
        Sink nodes to tag nodes index
    '''
    tags = tag_index.keys()
    num_tags = len(tags)
    num_sinks = len(sink_index)
    
    #This will prevent overlaps
    max_tag = len(tags)
    
    def edge_generator():
        '''Generates edges without using much more memory'''
        for tag in tags:
            seen = set()
            for sink in tag_index[tag]:
                for o_tag in sink_index[sink]:
                    if o_tag not in seen and tag != o_tag:
                        seen.add(o_tag)
                        yield (tag, o_tag)
                        
                new_id = sink + max_tag
                yield (tag, new_id)
    
    return num_tags, num_sinks, edge_generator()

def create_nxgraph(edges):
    '''Creates a graph object from iGraphs library'''
    return nx.DiGraph(edges)