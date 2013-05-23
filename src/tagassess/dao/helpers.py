# -*- coding: utf8
'''
Contains helpers classes for filtering data stored on DBs.
'''
from __future__ import division, print_function

class FilteredUserItemAnnotations(object):
    '''
    Auxiliary class which mocks annotation database by iterating over the old
    one and filtering out annotations based on items selected for removal.
    
    Arguments
    ---------
    user_item_pairs : dict of user to items
        User item pairs to be filtered out
    '''

    def __init__(self, user_item_pairs):
        self.user_item_pairs = user_item_pairs
        
    def annotations(self, annotations_it):
        '''
        Generates new annotations based on the original iterator to annotations
        
        Arguments
        ---------
        annotations_it : iterator to annotations
        '''
        
        for annotation in annotations_it:
            user = annotation['user']
            item = annotation['item']
            
            if user in self.user_item_pairs and \
                    item in self.user_item_pairs[user]:
                continue
            
            yield annotation