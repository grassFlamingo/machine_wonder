import unittest
from imageLoader import ImageLoader

class TestImageLoader(unittest.TestCase):
    iLoader = ImageLoader(
        "/media/aliy/DATA/STUDY/ComputerScience/DataSet/someOemos/bag",
        "/media/aliy/DATA/STUDY/ComputerScience/DataSet/someOemos/man"
    )

    def test_disk(self):
        for p,n in self.iLoader.disk_iteration_random(2,limit=4):
            self.assertEqual(len(p), 2)
            self.assertEqual(len(n), 2)
            for imgp, imgn in zip(p,n):
                self.assertEqual(imgp.shape, (224, 224,3))
                self.assertEqual(imgn.shape, (224, 224,3))

    def test_quick(self):
        for img, lab in self.iLoader.quick_iteration(2,2,limit=4):
            self.assertEqual(img.shape, (2,224,224,3))
            self.assertEqual(lab.shape, (2,2))

