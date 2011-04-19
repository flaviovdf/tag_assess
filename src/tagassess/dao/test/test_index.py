# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=C0103

from __future__ import print_function, division

from tagassess import data_parser
from tagassess import index_creator
from tagassess import test
from tagassess.dao.index import IndexReader
from tagassess.dao.index import IndexWriter

import os
import tempfile
import unittest

class TestIndexWriterReader(unittest.TestCase):
    '''Tests for basic reading and writing from PyTables files'''

    def setUp(self):
        self.h5_file = tempfile.mktemp('testw.h5')

    def tearDown(self):
        os.remove(self.h5_file)
        
    def test_read_write(self):
        parser = data_parser.Parser()
        annotations = []
        with open(test.BIBSONOMY_FILE) as in_f:
            for annot in parser.iparse(in_f, data_parser.bibsonomy_parser):
                annotations.append(annot)
        
        
        written_list = []
        written_set = set()
        with IndexWriter(self.h5_file) as index_writer:
            index_writer.create_table('i')
            for i in index_creator.create_metrics_index(annotations):
                index_writer.write(i)
                written_list.append(i)
                written_set.add(i)
        
        with IndexReader(self.h5_file) as index_reader:
            indices = [i for i in index_reader.iterate('i')]
            self.assertEquals(indices, written_list)
            self.assertEquals(set(indices), written_set)
        