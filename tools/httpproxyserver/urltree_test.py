import unittest
import urltree


class TestUrlTree(unittest.TestCase):
    def test_tree(self):
        gTree = urltree.URLTree()

        gTree.push_url("pl.music.163.com")
        gTree.push_url("clientlog3.music.163.com")
        gTree.push_url("*.abcd.com")

        self.assertTrue("xxx.abcd.com" in gTree)
        self.assertTrue("pl.music.163.com" in gTree)



        self.assertFalse("j.pl.music.163.com" in gTree)
        self.assertFalse("music.163.com" in gTree)


        