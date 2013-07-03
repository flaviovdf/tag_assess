# -*- coding: utf8
'''Probability based on pre-computed values'''

cimport base

import glob
import os
import tables

cdef class PrecomputedEstimator(base.ProbabilityEstimator):
    
    def __init__(self, probabilities_folder):
        user_files = os.path.join(probabilities_folder, 'user-*.h5')
        self.users_fpaths = glob.glob(user_files)
        self.user_to_piu = {}
        self.user_to_pitu = {}
        self.user_to_tags = {}
        self.user_to_gamma = {}
        
        for user_fpath in self.users_fpaths:
            user_id = int(user_fpath.split('-')[-1].split('.')[0])
            
            h5file = tables.openFile(user_fpath, mode='r')
            
            piu = h5file.getNode(h5file.root, 'piu').read()
            self.user_to_piu[user_id] = piu
            
            child_nodes = h5file.iterNodes(h5file.root)
            
            self.user_to_tags[user_id] = set()
            for child_node in child_nodes:
                if 'pitu' in child_node.name:
                    tag_id = int(child_node.name.split('_')[-1])
                    self.user_to_pitu[user_id, tag_id] = child_node.read()
                    self.user_to_tags[user_id].add(tag_id)
                        
            h5file.close()
            
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user(self, 
            int user, np.ndarray[np.int_t, ndim=1] gamma_items):
        return self.user_to_piu[user]
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_user_tag(self,
            int user, int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        return self.user_to_pitu[user, tag]
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items_given_tag(self,
            int tag, np.ndarray[np.int_t, ndim=1] gamma_items):
        return None
    
    cpdef np.ndarray[np.float_t, ndim=1] prob_items(self,
            np.ndarray[np.int_t, ndim=1] gamma_items):
        return None
    
    def tags_for_user(self, user):
        return self.user_to_tags[user]
    
    def gamma_for_user(self, user):
        return self.user_to_gamma[user]
