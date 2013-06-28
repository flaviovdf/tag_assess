# -*- coding: utf8
'''Probability based on pre-computed values'''

import glob
import os
import tables

class PrecomputedEstimator():
    
    def __init__(self, probabilities_folder):
        user_files = os.path.join(probabilities_folder, 'user-*.h5')
        self.users_fpaths = glob.glob(user_files)
        self.user_ids = set()
        self.user_to_piu = {} 
        self.user_to_pitu = {}
        self.user_to_tags = {}
        
        for user_fpath in self.users_fpaths:
            user_id = int(user_fpath.split('-')[-1].split('.')[0])
            
            h5file = tables.openFile(user_fpath, mode='r')
            
            gamma = h5file.getNode(h5file.root, 'gamma').read()
            self.user_to_piu[user_id].update(gamma)
            
            child_nodes = h5file.iterNodes(h5file.root)
            
            self.user_to_tags[user_id] = set()
            for child_node in child_nodes:
                if 'pitu' in child_node.name:
                    tag_id = int(child_node.name.split('_')[-1])
                    self.user_to_pitu[user_id, tag_id] = child_node.read()
                    self.user_to_tags[user_id].add(tag_id)
                        
            self.user_to_piu[user_id].update(gamma)
            
            h5file.close()
            
    def prob_items_given_user(self, user):
        return self.user_to_piu[user]

    def prob_items_given_user_tag(self, user, tag):
        return self.user_to_pitu[user, tag]
    
    def tags_for_user(self, user):
        return self.user_to_tags[user]