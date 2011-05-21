# -*- coding: utf8
'''Functions for creating tag graphs based on co-occurrence in items or users'''

from __future__ import division, print_function

from igraph import Graph 
from tagassess import index_creator
from tagassess.dao.annotations import AnnotReader

def extract_indexes_from_file(fpath, table, use=2):
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
    opts = {1:'user', 2:'item'}
    create_for = opts[use]
    
    with AnnotReader(fpath) as annotation_reader:
        iterator = annotation_reader.iterate(table)
        index = index_creator.create_occurrence_index(iterator, 
                                                      'tag', create_for)
    return index

def edge_list(index_for_tag_edges, uniq=False):
    '''
    Returns the edge list for the navigational graph.
    
    Arguments
    ---------
    index_for_tag_edges: dict (int -> list) 
        An index where the values are tag lists. These tags will
        be connected for the 'center' of the graph
    uniq: bool
        Indicates if ids in indices are already unique, that is no 
        tag, item and user shares the same id.
    '''
    edge_set = set()
    tag_nodes = set()
    tag_id_space = len(index_for_tag_edges)
    for tag1 in xrange(tag_id_space - 1):
        sinks_with_t1 = index_for_tag_edges[tag1]
        for tag2 in xrange(tag1 + 1, tag_id_space):
            sinks_with_t2 = index_for_tag_edges[tag2]
            
            if not sinks_with_t1.isdisjoint(sinks_with_t2):
                edge1 = (tag1, tag2)
                edge2 = (tag2, tag1)
                
                tag_nodes.add(tag1)
                tag_nodes.add(tag2)
                
                edge_set.add(edge1)
                edge_set.add(edge2)
            
    max_tag = 0
    if not uniq:
        #This will prevent overlaps        
        max_tag = tag_id_space
    
    sink_nodes = {}
    for tag in index_for_tag_edges:
        for item in index_for_tag_edges[tag]:
            new_item_id = max_tag + item
            sink_nodes[new_item_id] = item
            edge_set.add((tag, new_item_id))
    
    edges = []
    for source, dest in sorted(edge_set):
        edges.append((source, dest))
    
    return tag_nodes, sink_nodes, edges

def create_igraph(index_for_tag_edges, uniq=False):
    '''Creates a graph object from iGraphs library'''
    tag_nodes, sink_nodes, edges = \
     edge_list(index_for_tag_edges, uniq)
    return tag_nodes, sink_nodes, _create_igraph(tag_nodes, sink_nodes, edges)
    
def _create_igraph(tag_nodes, sink_nodes, edges):
    '''Creates a graph object from iGraphs library'''
    num_nodes = len(tag_nodes) + len(sink_nodes)
    return Graph(n=num_nodes, edges=edges, directed=True)