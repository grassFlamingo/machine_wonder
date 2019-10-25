import unittest

from dirimageloader import ImageLoader


class TestImageLoader(unittest.TestCase):
    iLoader = ImageLoader("/home/guest/DataSets/cat-and-dog/test_set")

    def test_walk(self):
        names = self.iLoader.name_of_classes()
        self.assertIn("cats", names)
        self.assertIn("dogs", names)
        self.assertEqual(self.iLoader.num_of_classes(), 2)

    def test_disk(self):
        for p,n in self.iLoader.disk_iteration(2, maxcount=4):
            self.assertEqual(len(p), 2)
            self.assertEqual(len(n), 2)
            for imgp, labs in zip(p,n):
                self.assertEqual(imgp.shape, (224, 224,3))
                self.assertLess(labs, self.iLoader.num_of_classes())
                self.assertGreaterEqual(labs, 0)

    def test_disk2(self):
        for p,n in self.iLoader.quick_iteration(2, 2, maxcount=4, outformat="NCHW"):
            self.assertEqual(len(p), 2)
            self.assertEqual(len(n), 2)
            for img, _ in zip(p,n):
                self.assertEqual(img.shape, (3, 224, 224))

    def test_quick(self):
        
        for i, img, lab in self.iLoader.quick_iteration(2,2, isenum=True, maxcount=4, useonehot=True, outformat="NHWC"):
            
            self.assertEqual(img.shape, (2,224,224,3))
            self.assertEqual(lab.shape, (2,2))
            self.assertLess(i, 2)
