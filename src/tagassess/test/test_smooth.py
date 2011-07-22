# -*- coding: utf8
#pylint: disable-msg=C0301
#pylint: disable-msg=C0111
from __future__ import division, print_function

from tagassess.test import PyCyUnit

class TestSmooth(PyCyUnit):

    def get_module_to_test(self):
        from tagassess import smooth
        return smooth

    def test_jm(self):
        self.assertEquals(self.mod_under_test.jelinek_mercer(10, 100, 20, 500, 0.6), 0.064)
        self.assertEquals(self.mod_under_test.jelinek_mercer(10, 100, 20, 500, 0.5), 0.07)
        self.assertEquals(self.mod_under_test.jelinek_mercer(1, 1, 1, 1, 1), 1)

    def test_bayes(self):
        self.assertAlmostEquals(self.mod_under_test.bayes(10, 100, 20, 500, 0.1), .0999, 4)
        self.assertAlmostEquals(self.mod_under_test.bayes(10, 100, 20, 500, 0.5), .0997, 4)
        self.assertEquals(self.mod_under_test.bayes(1, 1, 1, 1, 1), 1)
        
        self.assertAlmostEquals(self.mod_under_test.bayes(0, 100, 20, 500, 0.1), 0.000999000999 * (20/500))
