# -*- coding: utf8
'''Code for dealing with tag clouds'''

#TODO: Add a graph based cloud

from __future__ import division, print_function

from collections import defaultdict

import abc
import heapq

class BaseCloud(object):
    '''
    This class represents a tag cloud. It can be used
    to compute various tag cloud related metrics. This
    is an abstract class with the tag cloud metrics, different
    implementations will value tags differently inside the cloud.
    '''
    
    __metaclass__ = abc.ABCMeta
    
    def __init__(self, annotations, cloud_size = 20):
        '''
        Constructs a new cloud initialized with the top tags
        
        Arguments
        ---------
        annotations: iterable
            The annotations to consider
            
        cloud_size: int
            Determines the number of tags in the cloud
        '''
        
        tag_to_item = defaultdict(set)
        item_to_tag = defaultdict(set)
        
        for annotation in annotations:
            tag = annotation['tag']
            item = annotation['item']
            
            tag_to_item[tag].add(item)
            item_to_tag[item].add(tag)

            #For subclassess
            self._initialize(annotation)

        self.tag_to_item = tag_to_item
        self.item_to_tag = item_to_tag
        self.cloud_size = cloud_size
        
        #Initiate cloud with top tags
        self.current_cloud = None
        self.current_query = None
        self.update(None)

    def _initialize(self, annotation):
        '''
        Subclasses can implement this in order
        to update any internal data structures for 
        ranking tags later. This is called when the
        object is being initialized for each new annotation.
        '''
        pass

    def update(self, query):
        '''
        Set's a new query to create cloud
        
        Arguments
        ---------
        query: collection of tag ids
            The new query to fetch items with AND search
        '''
        query_result = set()
        if not query:
            possible_tags = self.tag_to_item.keys()
            query_result.update(self.item_to_tag.keys())
        else:
            #Simple and possibly inefficient AND search
            for term in query:
                if len(query_result) == 0:
                    query_result.update(self.tag_to_item[term])
                else:
                    query_result.intersection_update(self.tag_to_item[term])
                    
            #Building cloud
            possible_tags = set()
            for item in query_result:
                for tag in self.item_to_tag[item]:
                    possible_tags.add(tag)
        
        self.current_query = query
        self.current_cloud = self._top_tags(possible_tags, query_result)

    @abc.abstractmethod
    def _top_tags(self, tags, query_result):
        '''
        Abstract method for sorting tags based on
        the current return value of the query.
        
        Arguments
        ---------
        tags: collection of tag ids
            Tag ids to sort
        query_results: collection of item ids
            Current items returned by query
        '''
        pass

    def _precision(self, tags, items):
        '''Auxiliary method for computing precision'''
        
        reachable = set()
        for tag in tags:
            reachable.update(self.tag_to_item[tag])
            
        intersect_tag_relevant = reachable.intersection(items)
        return len(intersect_tag_relevant) / len(reachable)
        
    def precision(self, relevant_items):
        '''
        Precision for all tags in the cloud
        
        Arguments
        ---------
        relevant_items: collection of item ids
            Items to compute precision to
        '''
        return self._precision(self.current_cloud, relevant_items)
 
    def _recall(self, tags, items):
        '''Auxiliary method for computing recall'''
        
        reachable = set()
        for tag in tags:
            reachable.update(self.tag_to_item[tag])
            
        intersect_tag_relevant = reachable.intersection(items) 
        return len(intersect_tag_relevant) / len(items) 
    
    def recall(self, relevant_items):
        '''
        Recall for all tags in the cloud
        
        Arguments
        ---------
        relevant_items: collection of item ids
            Items to compute recall to
        '''
        return self._recall(self.current_cloud, relevant_items)
    
    def coverage(self, tags_to_cover):
        '''
        Summarizes cloud according to coverage
        
        Arguments
        ---------
        tags_to_cover: set of tag ids
            Tags to compute coverage to
        '''
        
        intersect_coverage = self.current_cloud.intersection(tags_to_cover)
        return  len(intersect_coverage) / len(tags_to_cover)
    
    def query_precision(self, relevant_items):
        '''
        Returns the precision of the current query

        Arguments
        ---------
        relevant_items: collection of item ids
            Items to compute precision to
        '''
        if not self.current_query:
            return 0.0
        
        return self._precision(self.current_query, relevant_items)
    
    def query_recall(self, relevant_items):
        '''
        Returns the precision of the current query
        
        Arguments
        ---------
        relevant_items: collection of item ids
            Items to compute recall to
            
        '''
        if not self.current_query:
            return 0.0        
        
        return self._recall(self.current_query, relevant_items)
    
    def __len__(self):
        return len(self.current_cloud)

class PreComputedValuesCloud(BaseCloud):
    '''
    This class creates a tag cloud with pre-computed
    values for the tags.
    '''
    
    def __init__(self, annotations, tag_value_map, 
                 cloud_size = 20):
        '''
        Constructs a new cloud initialized with the top tags
        
        Arguments
        ---------
        annotations: iterable
            The annotations to consider
            
        tag_value_map:
            A dict with the value of each tag
        
        cloud_size: int
            Determines the number of tags in the cloud
        '''
        #By Java standards, this is really ugly. Not sure if by python.
        self.tag_value_map = tag_value_map
        self.sort_by = lambda tag: self.tag_value_map[tag]
        super(PreComputedValuesCloud, self).__init__(annotations, cloud_size)
    
    def _top_tags(self, tags, query_result):
        return set(heapq.nlargest(self.cloud_size, tags, 
                                  key=self.sort_by))

class TFIDFCloud(BaseCloud):
    '''This class creates a tag cloud with based on TF IDF'''
    
    def __init__(self, annotations, cloud_size = 20, heuristic = 'tf-idf'):
        '''
        Constructs a new cloud initialized with the top tags
        
        Arguments
        ---------
        annotations: iterable
            The annotations to consider
            
        cloud_size: int
            Determines the number of tags in the cloud
            
        heuristic: str
            Which heuristic to use: 
                * 'tf-idf' for TF IDF
                * 'tf' for TF only
                * 'idf' for IDF only
                * 'inv-idf' for the inverse of IDF (equal to popularity)
        '''
        
        possible_heuristics = ['tf', 'tf-idf', 'idf', 'inv-idf']
        if heuristic not in possible_heuristics:
            raise ValueError('Unknown heuristic, please choose from: %s '
                             %' '.join(possible_heuristics))
        
        
        self.tag_user_freq = defaultdict(int)
        self.tag_col_freq = defaultdict(int)
        self.heuristic = heuristic
        
        super(TFIDFCloud, self).__init__(annotations, cloud_size)
        
        
    def _initialize(self, annotation):
        tag = annotation['tag']
        item = annotation['item']

        self.tag_col_freq[tag] += 1
        self.tag_user_freq[tag, item] += 1
        
    def _top_tags(self, tags, query_result):
        value_map = defaultdict(int)
        
        #Since we are only interest in rank, we don't really compute IDF
        #We use popularity or reverse popularity
        if self.heuristic == 'inv-idf':
            value_map = self.tag_col_freq
            
        elif self.heuristic == 'idf':
            for tag in tags:
                value_map[tag] = 1.0 / self.tag_col_freq[tag]
                
        else:
            #TF score
            for item in query_result:
                for tag in tags:
                    value_map[tag] += self.tag_user_freq[tag, item]
            
            #Consider idf
            if self.heuristic == 'tf-idf':
                for tag in value_map:
                    value_map[tag] /= self.tag_col_freq[tag]
        
        sort_by = lambda tag: value_map[tag]
        return set(heapq.nlargest(self.cloud_size, tags, 
                                  key=sort_by))