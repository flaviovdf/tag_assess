# -*- coding: utf8
'''Functions for creating tag graphs based on co-occurrence in items or users'''

from __future__ import division, print_function

from igraph import Graph 
from itertools import permutations
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
        base_index = index_creator.create_occurrence_index(iterator, 
                                                           create_for, 'tag')
        
        iterator = annotation_reader.iterate(table)
        tag_to_item_index = index_creator.create_occurrence_index(iterator, 
                                                                  'tag', 'item')
    return base_index, tag_to_item_index

def edge_list(index_for_tag_edges, tag_to_items_index, uniq=True):
    '''
    Returns the edge list for the navigational graph.
    
    Arguments
    ---------
    index_for_tag_edges: dict (int -> list) 
        An index where the values are tag lists. These tags will
        be connected for the 'center' of the graph
    tag_to_items_index: dict (int -> list)
        An index where keys are tags and values are items. These items
        will be connected with and outgoing edge from each tag
    uniq: bool
        Indicates if ids in indices are already unique, that is no 
        tag, item and user shares the same id.
    '''
    edge_set = set()
    for key in index_for_tag_edges:
        edge_set.update(permutations(index_for_tag_edges[key], 2))
    
    max_tag = 0
    if not uniq:
        #We need to find the max tag in order to add items.
        #Tag and items are ints with overlaps, this will mess up the graph
        max_tag = 0
        for vertex1, vertex2 in edge_set:
            aux = max(vertex1, vertex2)
            if aux >= max_tag:
                max_tag = aux
            
        #This will prevent overlaps        
        max_tag += 1
        
    for tag in tag_to_items_index:
        for item in tag_to_items_index[tag]:
            new_item_id = max_tag + item
            edge_set.add((tag, new_item_id))
    
    edges = []
    nodes = set()
    for source, dest in sorted(edge_set):
        nodes.add(source)
        nodes.add(dest)
        edges.append((source, dest))
    
    return nodes, edges

def create_igraph(index_for_tag_edges, tag_to_items_index, uniq=True):
    '''Creates a graph object from iGraphs library'''
    nodes, edges = edge_list(index_for_tag_edges, tag_to_items_index, uniq)
    return Graph(n=len(nodes), edges=edges, directed=True)