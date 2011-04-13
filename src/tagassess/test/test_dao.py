# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=C0103

'''Tests for the dao module'''

from __future__ import print_function, division

from tagassess import dao
from tagassess import data_parser
from tagassess import test

import os
import tempfile
import unittest

class TestAnnotWriterReader(unittest.TestCase):
    '''Tests for basic reading and writting from h5 files'''

    def setUp(self):
        self.h5_file = tempfile.mktemp('testw.h5')

    def tearDown(self):
        os.remove(self.h5_file)

    def test_create_and_write(self):
        '''
        This simple test writes annotations to annotation
        file and read them back. Comparing if both
        are equal.
        '''

        parser = data_parser.Parser()
        written = []
        with open(test.BIBSONOMY_FILE) as in_f:
            with dao.AnnotWriter(self.h5_file) as writer:
                writer.create_table('bibs')
                for annot in parser.iparse(in_f, data_parser.bibsonomy_parser):
                    written.append(annot)
                    writer.write(annot)

        with dao.AnnotReader(self.h5_file, 'bibs') as reader:
            read = [annot for annot in reader]
            self.assertEquals(read, written)

if __name__ == "__main__":
    unittest.main()