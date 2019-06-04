import numpy as np
import cv2
import imageio
import os
import itertools

class ImageLoader:
    def __init__(self, bagpath : str, manpath: str):
        """
        Input:
        - bagpath: a path in which all images are contains human
        - manpath: a path in which all images are backgrounds
        """
        self.bagpath = bagpath
        self.manpath = manpath

    def __loadimages(self, path, limit):
        imgPath = os.listdir(path)
        lenn = min(limit, len(imgPath))
        imgs = [resizeImageWithRandomCrop(imageio.imread(os.path.join(path, p)))
                    for p in imgPath[0:lenn]]
        imgs = np.asarray(imgs) / 255
        np.random.shuffle(imgs)
        return imgs

    def load_bag(self, limit=100):
        """
        load negative images (background) limited by limit
        """
        return self.__loadimages(self.bagpath, limit)
    
    def load_man(self, limit=100):
        """
        load positive images (human) limited by limit
        """
        return self.__loadimages(self.manpath, limit)


    def disk_iteration_random(self, batch=20, limit=None):
        """
        Randomly pick images from disk
        All return images are (224, 224)
        Input:
        - batch: how many images to pick per time
        - limit: total numbers, if None, this function won't stop
        Return:
        - imgP: pisitive images
        - imgN: negative images
        """
        imgsP = os.listdir(self.bagpath)
        imgsN = os.listdir(self.manpath)
        lenp, lenn = len(imgsP), len(imgsN)
        if limit:
            iteror = range(limit)
        else:
            iteror = itertools.count()
        for i in iteror:
            mp = np.random.randint(0,lenp, (batch,))
            mn = np.random.randint(0,lenn, (batch,))
            imgP, imgN = [], []
            for p, n in zip(mp, mn):
                rawp = imageio.imread(os.path.join(self.bagpath, imgsP[p]))
                rawn = imageio.imread(os.path.join(self.manpath, imgsN[n]))
                imgP.append(resizeImageWithRandomCrop(rawp))
                imgN.append(resizeImageWithRandomCrop(rawn))
            yield imgP, imgN

    def list_iteration(self, imgP:list, imgN:list, batch = 10):
        """
        Input:
        - imgP: Positive images (human in it)
        - imgN: Negative images (background only)
        Return:
        - image: a numpy.ndarray (N, H, W, C)
        - label: a numpy.ndarray (N,2) one hot vector 
                (1, 0) -> Positive
                (0, 1) -> Negative
        """
        imgP = np.asarray(imgP) / 255
        imgN = np.asarray(imgN) / 255

        labP = np.zeros((imgP.shape[0],2))
        labP[:,0] = 1;

        labN = np.zeros((imgP.shape[0],2))
        labN[:,1] = 1;

        img = np.concatenate([imgP, imgN], axis=0)
        lab = np.concatenate([labP, labN], axis=0)

        mask = np.arange(0, img.shape[0], 1)
        np.random.shuffle(mask)
        np.random.shuffle(mask)

        img = img[mask]
        lab = lab[mask]
        
        for i in range(0, mask.shape[0], batch):
            yield img[i:i+batch], lab[i:i+batch]

    def quick_iteration(self, diskbatch=20, listbatch=10, limit=None):
        """
        A combination of ```disk_iteration_random``` and ```list_iteration```
        """
        for imgp, imgn in self.disk_iteration_random(diskbatch, limit=limit):
            yield from self.list_iteration(imgp, imgn, listbatch)

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
        cpimg = img[:,a:a+W,:]
    return cv2.resize(cpimg, (224, 224), interpolation=cv2.INTER_LINEAR)


