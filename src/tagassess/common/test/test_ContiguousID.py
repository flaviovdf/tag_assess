# -*- coding: utf8
from __future__ import print_function, division

import unittest
import random

from tagassess.common import ContiguousID

class TestContiguousID(unittest.TestCase):

    def test_all(self):
        x = ContiguousID()
        self.assertEquals(x['a'], 0)
        self.assertEquals(x['b'], 1)
        self.assertEquals(x[0], 2)

        self.assertEquals(x['a'], 0)
        self.assertEquals(x[int], 3)
        
        new_keys = range(1, 200)
        random.shuffle(new_keys)
        iter_vals = iter(range(4, 204))
        for i in new_keys:
            self.assertEquals(x[i], next(iter_vals))
        
        for i, d in enumerate(sorted(x.itervalues())):
            self.assertEqual(i, d)
        
        self.assertEquals(203, len(x))

if __name__ == "__main__":
    unittest.main()