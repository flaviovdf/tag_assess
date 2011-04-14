# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=C0103

from __future__ import print_function, division

from tagassess import dao
from tagassess import data_parser
from tagassess import test

import os
import tempfile
import unittest

class TestAnnotWriterReader(unittest.TestCase):
    '''Tests for basic reading and writing from PyTables files'''

    def setUp(self):
        self.h5_file = tempfile.mktemp('testw.h5')

    def tearDown(self):
        os.remove(self.h5_file)

    def base_tfunc(self, fpath, parse_func):
        '''
        This simple test writes annotations to h5 file and reads them back. 
        Comparing if both are equal.
        '''
        
        parser = data_parser.Parser()
        written_list = []
        written_set = set()
        n_lines = 0
        with open(fpath) as in_f:
            with dao.AnnotWriter(self.h5_file) as writer:
                writer.create_table('bibs')
                for annot in parser.iparse(in_f, parse_func):
                    written_list.append(annot)
                    written_set.add(annot)
                    writer.write(annot)
                    n_lines += 1

        with dao.AnnotReader(self.h5_file, 'bibs') as reader:
            read_list = [annot for annot in reader]
            self.assertEquals(read_list, written_list)
            self.assertEquals(set(read_list), written_set)
            self.assertEquals(n_lines, len(written_list))
            self.assertEquals(n_lines, len(written_set))
            

    def test_create_and_write_bibsonomy(self):
        self.base_tfunc(test.BIBSONOMY_FILE, data_parser.bibsonomy_parser)

    def test_create_and_write_citeulike(self):
        self.base_tfunc(test.CITEULIKE_FILE, data_parser.citeulike_parser)

    def test_create_and_write_connotea(self):
        self.base_tfunc(test.CONNOTEA_FILE, data_parser.connotea_parser)

    def test_create_and_write_delicious(self):
        self.base_tfunc(test.DELICIOUS_FILE, data_parser.delicious_flickr_parser)

    def test_create_and_write_flickr(self):
        self.base_tfunc(test.FLICKR_FILE, data_parser.delicious_flickr_parser)

if __name__ == "__main__":
    unittest.main()