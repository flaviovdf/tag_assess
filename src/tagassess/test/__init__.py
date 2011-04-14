'''
Test functions for tagassess module. This
package also initializes the small files
used for testing.
'''

import os

#I don't know if this the best way to locate test files. But it works.
DATA_DIR = os.path.join(__path__[0], 'sample_data')
BIBSONOMY_FILE = os.path.join(DATA_DIR, 'bibsonomy.dat')
CITEULIKE_FILE = os.path.join(DATA_DIR, 'citeulike.dat')
CONNOTEA_FILE = os.path.join(DATA_DIR, 'connotea.dat')
DELICIOUS_FILE = os.path.join(DATA_DIR, 'delicious.dat')
FLICKR_FILE = os.path.join(DATA_DIR, 'flickr.dat')