'''
Test functions for tagassess module. This
package initializes the small files
used for testing and declares utility classes/functions.
'''

import os

#I don't know if this the best way to locate test files. But it works.
DATA_DIR = os.path.join(__path__[0], 'sample_data')
BIBSONOMY_FILE = os.path.join(DATA_DIR, 'bibsonomy.dat')
CITEULIKE_FILE = os.path.join(DATA_DIR, 'citeulike.dat')
CONNOTEA_FILE = os.path.join(DATA_DIR, 'connotea.dat')
DELICIOUS_FILE = os.path.join(DATA_DIR, 'delicious.dat')
FLICKR_FILE = os.path.join(DATA_DIR, 'flickr.dat')
SMALL_DEL_FILE = os.path.join(DATA_DIR, '10_annotations_delicious.dat')

#Class for sharing tests with Cython modules
import unittest

class PyCyUnit(unittest.TestCase):
    '''
    This is an abstract class used to share the same
    test between Python and Cython versions of a module.
    '''
    def get_module_to_eval(self, *args, **kwargs):
        '''Returns the module under test'''
        pass