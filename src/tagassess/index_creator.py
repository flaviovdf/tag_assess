# -*- coding: utf8
'''
Functions used to create indices and reverse lists. 
'''
from __future__ import division, print_function

from collections import defaultdict

def create_occurrence_index(annotation_it, from_, dest):
    '''
    Creates reversed occurrence indices. Basically, we
    can index the items or users which contain a given tag; or
    the items or tags which a user annotated; or the tags and users
    associated with items.
    
    Arguments
    ---------
    annotation_it: any iterable
        the annotations to process
    from_: str 
        the key of the index {'tag', 'item', 'user'}
    dest: str
        the lists to create. e.g from tags to items for a reverse tag index.
        {'tag', 'item', 'user'}
    
    Returns
    -------
    A dict with the index
        
    See also
    --------
    tagassess.dao.Annotation
    '''
    occurence_index = defaultdict(set)
    for annot in annotation_it:
        from_id = annot[from_]
        dest_id = annot[dest]
        occurence_index[from_id].add(dest_id)
        
    return occurence_index

def create_double_occurrence_index(annotation_it, from_, dest):
    '''
    Creates double way occurrence indices. This is the same
    as two occurrence indexes in each direction. The first
    tuple returned has a index : from -> dest, while the
    second is dest -> from.
    
    Arguments
    ---------
    annotation_it: any iterable
        the annotations to process
    from_: str 
        the key of the index {'tag', 'item', 'user'}
    dest: str
        the lists to create. e.g from tags to items for a reverse tag index.
        {'tag', 'item', 'user'}
    
    Returns
    -------
    A dict with the index
        
    See also
    --------
    tagassess.dao.Annotation
    '''
    from_to_dest = defaultdict(set)
    dest_to_from = defaultdict(set)
    for annot in annotation_it:
        from_id = annot[from_]
        dest_id = annot[dest]
        from_to_dest[from_id].add(dest_id)
        dest_to_from[dest_id].add(from_id)
        
    return (from_to_dest, dest_to_from)

def create_metrics_index(annotation_it, from_, dest):
    '''
    Creates a simple metrics index. Basically it counts
    the individual occurrences of `from_` and `dest` as 
    also the occurrences of both. 
    
    Arguments
    ---------
    annotation_it: any iterable
        the annotations to process
    from_: str
        the key of the index {'tag', 'item', 'user'}
    dest: str
        the lists to create. e.g from tags to items for a reverse tag index.
        {'tag', 'item', 'user'}
    
    Returns
    -------
    A tuple of three dictionaries. The first with the frequencies of both
    `dest` and `from_` together, a second with the global `from_` frequencies and a third
    with the global `dest`.
    
    See also
    --------
    tagassess.dao.Annotation
    '''
    from_dest_frequencies = defaultdict(lambda: defaultdict(int))
    collection_from_frequency = defaultdict(int)
    collection_dest_frequency = defaultdict(int)
    
    for annot in annotation_it:
        from_id = annot[from_]
        dest_id = annot[dest]
        
        from_dest_frequencies[from_id][dest_id] += 1
        collection_from_frequency[from_id] += 1
        collection_dest_frequency[dest_id] += 1
    
    return (from_dest_frequencies, collection_from_frequency, 
            collection_dest_frequency)