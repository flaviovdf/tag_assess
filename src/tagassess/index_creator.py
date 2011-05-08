# -*- coding: utf8
#pylint: disable-msg=W0401
#pylint: disable-msg=W0614
'''
Functions used to create indices and reverse lists. 
'''
from __future__ import division, print_function

from collections import defaultdict
from tagassess.dao.index import IndexInf

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
    methods = {'tag':lambda annot: annot.get_tag(),
               'item':lambda annot: annot.get_item(),
               'user':lambda annot: annot.get_user()}
    
    occurence_index = defaultdict(set)
    for annot in annotation_it:
        from_id = methods[from_](annot)
        dest_id = methods[dest](annot)
        occurence_index[from_id].add(dest_id)
        
    return occurence_index

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
    methods = {'tag':lambda annot: annot.get_tag(),
               'item':lambda annot: annot.get_item(),
               'user':lambda annot: annot.get_user()}
        
    from_dest_frequencies = defaultdict(lambda: defaultdict(int))
    collection_from_frequency = defaultdict(int)
    collection_dest_frequency = defaultdict(int)
    
    for annot in annotation_it:
        from_id = methods[from_](annot)
        dest_id = methods[dest](annot)
        
        from_dest_frequencies[from_id][dest_id] += 1
        collection_from_frequency[from_id] += 1
        collection_dest_frequency[dest_id] += 1
    
    return (from_dest_frequencies, collection_from_frequency, 
            collection_dest_frequency)

def metric_index_to_dao(term_frequencies, collection_frequency):
    '''
    Converts the term frequencies and collection frequencies to
    dao objects. This is to be called if the metrics index does
    not fit in main memory
    
    Arguments
    ---------
    term_frequencies: defaultdict(lambda: defaudict(list))
    collection_frequency: dict to integer
    '''
    return_index = []
    for posterior in term_frequencies:
        for tag in term_frequencies[posterior]:
            inf = IndexInf(posterior, tag, 
                           term_frequencies[posterior][tag], 
                           collection_frequency[tag])
            return_index.append(inf)
    
    return return_index