# -*- coding: utf8
#pylint: disable-msg=W0401
#pylint: disable-msg=W0614
'''
Functions used to create indices and reverse lists. 
'''
from __future__ import division, print_function

from tagassess.dao.index import IndexInf
from collections import defaultdict

def create_metrics_index(annotation_it, user=False):
    '''
    Creates a simple metrics index which contains the:
    ITEM TAG TAG_FREQUENCY_IN_ITEM TAG_FREQUENCY_COLLECTION
    
    Arguments
    ---------
    annotation_it: any iterable
        the annotations to process
    user: bool 
        indicates if index should be computed for users
        instead of items
    
    See also
    --------
    tagassess.dao.Annotation
    '''
    term_frequencies = defaultdict(lambda: defaultdict(int))
    collection_frequency = defaultdict(int)
    
    for annot in annotation_it:
        if user:
            posterior = annot.get_user()
        else:
            posterior = annot.get_item()
            
        tag = annot.get_tag()
        
        term_frequencies[posterior][tag] += 1
        collection_frequency[tag] += 1
    
    return_index = []
    for posterior in term_frequencies:
        for tag in term_frequencies[posterior]:
            inf = IndexInf(posterior, tag, 
                           term_frequencies[posterior][tag], 
                           collection_frequency[tag])
            return_index.append(inf)
    
    return return_index