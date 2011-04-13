# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
#pylint: disable-msg=C0103

from __future__ import print_function, division

from tagassess import data_parser
from tagassess import test

import StringIO
import time
import unittest

DELICIOUS_LINE1 = '2003-01-01 01:00:00     2384    125497  tinker'
DELICIOUS_LINE2 = '2011-02-17 11:10:20     2384    674518  hardware'
DELICIOUS_LINE3 = '2003-01-01 01:00:00     1       674518  hardware pc'
DELICIOUS_LINE4 = '2003-01-01 01:00:00     3       674518  tinker'
DELICIOUS_LINE5 = '2005-03-03 01:00:00     8718    111111  bala'
DELICIOUS_LINE6 = '2005-03-03 01:00:00     125497    2384  bala'
DELICIOUS_LINE7 = '2005-03-03 01:00:00     4    2384  125497'

CONNOTEA_LINE1 = '286ebecbce9fe3d99432c349fb2851c3|timo|2004-12-09T18:37:12Z|review'
CONNOTEA_LINE2 = '1234555asd122d432c349fb285aas1c3|sand|2004-12-09T18:37:12Z|long-term potentiation'

BIBSONOMY_LINE1 = '0       boomerang       7       1       2005-12-15 19:31:50'
BIBSONOMY_LINE2 = '2       shop    6       1       2005-12-15 19:31:50'

CITEUL_LINE1 = '4184140|aeb5429a4c20c7360579f53366633144|2009-03-16 17:58:33.792331+00|flavivirus'
CITEUL_LINE2 = '2820125|e4fc89df8b47cf4eaede9b9f1620c57f|2009-03-16 18:00:04.165438+00|partial-order-reduction'

def convert_time(t):
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(t))

class TestParseFuncs(unittest.TestCase):
    '''
    Basic tests for line parser functions
    '''

    def test_delicious_flickr(self):
        user, item, tag, date = data_parser.delicious_flickr_parser(DELICIOUS_LINE1)

        self.assertEqual(2384, user)
        self.assertEqual(125497, item)
        self.assertEqual('tinker', tag)
        self.assertEqual('2003-01-01 01:00:00', convert_time(date))

        user, item, tag, date = data_parser.delicious_flickr_parser(DELICIOUS_LINE2)
        self.assertEqual(2384, user)
        self.assertEqual(674518, item)
        self.assertEqual('hardware', tag)
        self.assertEqual('2011-02-17 11:10:20', convert_time(date))

        user, item, tag, date = data_parser.delicious_flickr_parser(DELICIOUS_LINE3)
        self.assertEqual(1, user)
        self.assertEqual(674518, item)
        self.assertEqual('hardware pc', tag)
        self.assertEqual('2003-01-01 01:00:00', convert_time(date))


    def test_connotea(self):
        user, item, tag, date = data_parser.connotea_parser(CONNOTEA_LINE1)

        self.assertEqual('timo', user)
        self.assertEqual('286ebecbce9fe3d99432c349fb2851c3', item)
        self.assertEqual('review', tag)
        self.assertEqual('2004-12-09 18:37:12', convert_time(date))

        user, item, tag, date = data_parser.connotea_parser(CONNOTEA_LINE2)
        self.assertEqual('sand', user)
        self.assertEqual('1234555asd122d432c349fb285aas1c3', item)
        self.assertEqual('long-term potentiation', tag)
        self.assertEqual('2004-12-09 18:37:12', convert_time(date))

    def test_bibsonomy(self):
        user, item, tag, date = data_parser.bibsonomy_parser(BIBSONOMY_LINE1)

        self.assertEqual(0, user)
        self.assertEqual(7, item)
        self.assertEqual('boomerang', tag)
        self.assertEqual('2005-12-15 19:31:50', convert_time(date))

        user, item, tag, date = data_parser.bibsonomy_parser(BIBSONOMY_LINE2)
        self.assertEqual(2, user)
        self.assertEqual(6, item)
        self.assertEqual('shop', tag)
        self.assertEqual('2005-12-15 19:31:50', convert_time(date))

    def test_citeul(self):
        user, item, tag, date = data_parser.citeulike_parser(CITEUL_LINE1)

        self.assertEqual('aeb5429a4c20c7360579f53366633144', user)
        self.assertEqual(4184140, item)
        self.assertEqual('flavivirus', tag)
        self.assertEqual('2009-03-16 17:58:33', convert_time(date))

        user, item, tag, date = data_parser.citeulike_parser(CITEUL_LINE2)
        self.assertEqual('e4fc89df8b47cf4eaede9b9f1620c57f', user)
        self.assertEqual(2820125, item)
        self.assertEqual('partial-order-reduction', tag)
        self.assertEqual('2009-03-16 18:00:04', convert_time(date))

class TestIParse(unittest.TestCase):

    def test_iparse(self):
        fakef = StringIO.StringIO()
        fakef.writelines([DELICIOUS_LINE1 + '\n',
                          DELICIOUS_LINE2 + '\n',
                          DELICIOUS_LINE3 + '\n',
                          DELICIOUS_LINE4 + '\n',
                          DELICIOUS_LINE5])
        fakef.seek(0)

        p = data_parser.Parser()
        annots = [a for a in p.iparse(fakef, data_parser.delicious_flickr_parser)]

        self.assertEqual(0, annots[0].get_user())
        self.assertEqual(0, annots[0].get_item())
        self.assertEqual(0, annots[0].get_tag())
        self.assertEqual('2003-01-01 01:00:00', convert_time(annots[0].get_date()))

        self.assertEqual(0, annots[1].get_user())
        self.assertEqual(1, annots[1].get_item())
        self.assertEqual(1, annots[1].get_tag())
        self.assertEqual('2011-02-17 11:10:20', convert_time(annots[1].get_date()))

        self.assertEqual(1, annots[2].get_user())
        self.assertEqual(1, annots[2].get_item())
        self.assertEqual(2, annots[2].get_tag())

        self.assertEqual(2, annots[3].get_user())
        self.assertEqual(1, annots[3].get_item())
        self.assertEqual(0, annots[3].get_tag())

        self.assertEqual(3, annots[4].get_user())
        self.assertEqual(2, annots[4].get_item())
        self.assertEqual(3, annots[4].get_tag())

    def test_iparse_idshare(self):
        fakef = StringIO.StringIO()
        fakef.writelines([DELICIOUS_LINE1 + '\n',
                          DELICIOUS_LINE2 + '\n',
                          DELICIOUS_LINE3])
        fakef.seek(0)

        p = data_parser.Parser(True)
        annots = [a for a in p.iparse(fakef, data_parser.delicious_flickr_parser)]

        self.assertEqual(0, annots[0].get_user())
        self.assertEqual(1, annots[0].get_item())
        self.assertEqual(2, annots[0].get_tag())

        self.assertEqual(0, annots[1].get_user())
        self.assertEqual(3, annots[1].get_item())
        self.assertEqual(4, annots[1].get_tag())

        self.assertEqual(5, annots[2].get_user())
        self.assertEqual(3, annots[2].get_item())
        self.assertEqual(6, annots[2].get_tag())

    def test_iparse_idshare_sametext(self):
        fakef = StringIO.StringIO()
        fakef.writelines([DELICIOUS_LINE1 + '\n',
                          DELICIOUS_LINE6 + '\n',
                          DELICIOUS_LINE7])
        fakef.seek(0)

        p = data_parser.Parser(True)
        annots = [a for a in p.iparse(fakef, data_parser.delicious_flickr_parser)]

        self.assertEqual(0, annots[0].get_user())
        self.assertEqual(1, annots[0].get_item())
        self.assertEqual(2, annots[0].get_tag())

        self.assertEqual(3, annots[1].get_user())
        self.assertEqual(4, annots[1].get_item())
        self.assertEqual(5, annots[1].get_tag())

        self.assertEqual(6, annots[2].get_user())
        self.assertEqual(4, annots[2].get_item())
        self.assertEqual(7, annots[2].get_tag())

    def test_with_file(self):
        p = data_parser.Parser(False)
        with open(test.BIBSONOMY_FILE) as f:
            annots = [a for a in p.iparse(f, data_parser.bibsonomy_parser)]
            self.assertEquals(10000, len(annots))

if __name__ == "__main__":
    unittest.main()