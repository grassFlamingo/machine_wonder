import numpy as np
import cv2
import imageio
import os
import itertools

__all__ = ["ImageLoader"]

G_IMAGE_SUFFIX = tuple("jpg,jpge,png".split(","))


class SubFolderWalkers:
    def __init__(self, path):
        self.path = path
        self.mimages = []
        for img in os.listdir(path):
            if not img.lower().endswith(G_IMAGE_SUFFIX):
                continue
            self.mimages.append(img)
        

    def __iter__(self):
        np.random.shuffle(self.mimages)
        self.itimg = iter(self.mimages)
        return self

    def __next__(self):
        return os.path.join(self.path, next(self.itimg))


class MainFolderWalkers:
    def __init__(self, mainpath):
        self.mainpath = mainpath

        self.subwalkers = []
        self.subpaths = []

        for cf in os.listdir(mainpath):
            abscf = os.path.join(mainpath, cf)
            if not os.path.isdir(abscf):
                continue
            self.subpaths.append(cf)
            self.subwalkers.append(SubFolderWalkers(abscf))

    def subpaths_names(self):
        return self.subpaths

    def __len__(self):
        return len(self.subpaths)

    def find_subpath_names(self, index):
        return self.subpaths[index]

    def walk_files(self):
        index = 0
        thepaths = [(i, iter(wk)) for i, wk in enumerate(self.subwalkers)]
        pathlen = len(thepaths)
        while pathlen > 0:
            index = index % pathlen
            try:
                currentp = thepaths[index]
                yield currentp[0], next(currentp[1])
                index = index + 1
            except StopIteration:
                del thepaths[index]
                pathlen = pathlen - 1


class ImageLoader:
    def __init__(self, mainpath: str):
        """
        Input:
        - mainpath: a folder where all folders in it are filled with same class images
            such as
            |- mainpath
            |----class1
            |------img1.xxx
            |------img2.xxx
            |------.....
            |----class2
        """
        self.mainwalker = MainFolderWalkers(mainpath)

    def num_of_classes(self) -> int:
        return len(self.mainwalker)

    def name_of_classes(self) -> list:
        return self.mainwalker.subpaths_names()

    def __loadimages(self, path, maxcount) -> np.ndarray:
        imgPath = os.listdir(path)
        lenn = min(maxcount, len(imgPath))
        imgs = [resizeImageWithRandomCrop(imageio.imread(os.path.join(path, p)))
                for p in imgPath[0:lenn]]
        imgs = np.asarray(imgs, dtype=np.float32) / 255
        np.random.shuffle(imgs)
        return imgs

    def disk_iteration(self, batch=20, maxcount=None):
        """
        Randomly pick images from disk
        All return images are (224, 224)
        Requires:
            - batch: how many images to pick per time
            - maxcount: total numbers, if None, this function will walk through all images
        Return:
            - images
            - labels
        """
        iteror = self.mainwalker.walk_files()
        if not maxcount is None:
            iteror = itertools.islice(iteror, 0, maxcount)
        imgs, labs = [], []

        for i, path in iteror:
            try:
                rawp = resizeImageWithRandomCrop(imageio.imread(path))
                imgs.append(rawp)
                labs.append(i)
            except Exception as e:
                print("cannot read image at", path, "due to", e)
            if len(labs) >= batch:
                yield imgs, labs
                imgs.clear()
                labs.clear()

    def list_iteration(
            self, imgs: list, labs: list, batch=10,
            useonehot: bool = False, outformat="NCHW"):
        """
        Input:
            - imgs: the images
            - labs: the labels
        Return:
            - image: a numpy.ndarray (N, H, W, C) if "NHWC" else (N, C, H, W)
            - label: a numpy.ndarray (N) if not use onehot else (N, num of classes)
        """
        imgs = np.asarray(imgs, dtype=np.float32) / 255
        labs = np.asarray(labs, dtype=np.int32)

        N, C = imgs.shape[0], self.num_of_classes()


        mask = np.arange(0, N, 1)
        np.random.shuffle(mask)

        imgs = imgs[mask]
        labs = labs[mask]

        if outformat == "NCHW":
            imgs = np.transpose(imgs, (0,3,1,2))
        elif outformat == "NHWC":
            pass
        else:
            raise ValueError("outformat only support NCHW or NHWC")

        for start in range(0, N, batch):
            end = max(start+batch, N)
            timg = imgs[start:end]
            tlab = labs[start:end]
            if useonehot:
                labz = np.zeros((end - start, C))
                for i, l in enumerate(tlab):
                    labz[i, l] = 1
                tlab = labz
            yield timg, tlab

    def quick_iteration(
            self, diskbatch: int, listbatch: int,
            isenum=False,
            maxcount=None, useonehot = False, outformat="NCHW"):
        """
        A combination of ```disk_iteration``` and ```list_iteration```
        Require:
            - diskbatch:
            - listbatch:
            - isenum(bool): if true return (i, image, label) else (iamge, label)
            - maxcount: max image to load
            - useonehot: whether to use onehot encoded label
            - outformat: only support ["NCHW", "NHWC"]
        Yield:
            - i: if isenum is true
            - images: 
            - labels:
        """
        assert diskbatch >= listbatch, "diskbatch must grater than listbatch"

        index = 0
        
        for imgs, labs in self.disk_iteration(diskbatch, maxcount=maxcount):
            liter = self.list_iteration(imgs, labs, listbatch, useonehot=useonehot, outformat=outformat)
            if isenum:
                for lit in liter:
                    yield (index, *lit)
                    index += 1
            else:
                yield from liter
                


def resizeImageWithRandomCrop(img: np.ndarray):
    W, H, _ = img.shape
    f = W / H
    if f - 1 < 1e-8:
        # square reshape simplly
        cpimg = img
    elif f > 1.0:
        # W > H
        a = np.random.randint(0, W-H)
        cpimg = img[a:a+H, :, :]
    else:
        # W < H
        a = np.random.randint(0, H-W)
        cpimg = img[:, a:a+W, :]
    return cv2.resize(cpimg, (224, 224), interpolation=cv2.INTER_LINEAR)
