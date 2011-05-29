# -*- coding: utf8
'''Functions for creating tag graphs based on co-occurrence in items or users'''

from __future__ import division, print_function

from igraph import Graph 
from tagassess import index_creator
from tagassess.dao.annotations import AnnotReader

def __extract_indexes_from_file(fpath, table, use=2):
    '''
    Creates indexes and sets needed.
    
    Arguments
    ---------
    fpath: str
        The path to the annotation file
    table: str
        The table to use
    use = int {1, 2}
        Indicates whether to use items or users:
            1: Items
            2: Users
    '''
    opts = {1:'item', 2:'user'}
    create_for = opts[use]
    
    with AnnotReader(fpath) as annotation_reader:
        iterator = annotation_reader.iterate(table)
        tag_index = index_creator.create_occurrence_index(iterator, 
                                                          'tag', create_for)
        iterator = annotation_reader.iterate(table)
        sink_index = index_creator.create_occurrence_index(iterator, 
                                                           create_for, 'tag')
    return tag_index, sink_index

def iedge_list(fpath, table, use=1, uniq=False):
    '''
    Returns the edge list for the navigational graph.
    
    Arguments
    ---------
    fpath: str
        The path to the annotation file
    table: str
        The table to use
    use = int {1, 2}
        Indicates whether to use items or users:
            1: Items
            2: Users
    uniq: bool
        Indicates if ids in indices are already unique, that is no 
        tag, sink and user shares the same id.
    '''
    tag_index, sink_index = __extract_indexes_from_file(fpath, table, use)
    tags = tag_index.keys()
    num_tags = len(tags)
    num_sinks = len(sink_index)
    
    max_tag = 0
    if not uniq:
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

def create_igraph(edges):
    '''Creates a graph object from iGraphs library'''
    return Graph(edges=edges, directed=True)