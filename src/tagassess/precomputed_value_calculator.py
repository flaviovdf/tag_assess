# -*- coding: utf8

'''Common functions to be called by scripts for experiments'''

from __future__ import division, print_function

from tagassess import entropy
from tagassess.index_creator import create_occurrence_index
from tagassess.stats.topk import kendall_tau_distance as dktau

import numpy as np

class PrecompValueCalculator(object):
    
    def __init__(self, estimator, annotation_it):
        self.est = estimator
        self.items_with_tag = dict((k, v) for k, v in 
            create_occurrence_index(annotation_it, 'tag', 'item').items())

    def calc_rho(self, tag, top_valued_items):
        top_valued_items_with_tag = []
        
        for i in range(top_valued_items.shape[0]):
            item_id = top_valued_items[i] #Top valued items is already sorted
            if item_id in self.items_with_tag[tag]:
                top_valued_items_with_tag.append(item_id)
        
        k = len(top_valued_items_with_tag)
        if k == 0:
            return 0
        else:
            return 1 / (1 + dktau(top_valued_items, top_valued_items_with_tag, 
                                  k, p=1))

    def tag_value_personalized(self, user, tags, return_rho_dkl=False):
        
        if not return_rho_dkl:
            return_val = np.ndarray(shape=(tags.shape[0],), dtype='d')
        else:
            return_val = np.ndarray(shape=(tags.shape[0], 3), dtype='d')
       
        for i in range(tags.shape[0]):
            tag = tags[i]
            vp_iu = self.est.prob_items_given_user(user)
            vp_itu = self.est.prob_items_given_user_tag(user, tag)
            rho = self.calc_rho(tag, vp_iu.argsort()[::-1])
            dkl = entropy.kullback_leiber_divergence(vp_itu, vp_iu)
            tag_val = rho * dkl
            
            if not return_rho_dkl:
                return_val[i] = tag_val
            else:
                return_val[i, 0] = rho
                return_val[i, 1] = dkl
                return_val[i, 2] = tag_val
        return return_val