import numpy as np
import scipy.io
import PIL.Image as Image
import os

import argparse

_args = argparse.ArgumentParser()

_args.add_argument("--inpath", "-i", type=str,
                   required=True, help="input path")
_args.add_argument("--output", "-o", type=str,
                   required=True, help="output path")
_args.add_argument("--format", "-f", type=str, default="NHWC",
                   help="format 'NCHW', N: number, C: channel, H: height, W: width")
_args.add_argument("--resize", "-r", type=str, default=None,
                   help="resize size width,height")
_args.add_argument("--suffix", "-s", type=str,
                   required=True, help="suffix of images")
args = _args.parse_args()

if not os.path.exists(args.inpath):
    print("input path not found", args.inpath)
    exit(0)

print("input folder:", args.inpath)

outfmt = str(args.format).upper()  # type: str

assert len(outfmt) == 4, "format must be 4 letters"

outdict = {"N": 0, "H": 1, "W": 2, "C": 3}

out = []
for file in sorted(os.listdir(args.inpath)):
    if not file.endswith(args.suffix):
        continue
    print("found image:", file)
    img = Image.open(os.path.join(args.inpath, file))
    if args.resize:  # type: str
        w, h = args.resize.split(",")
        w, h = int(w), int(h)
        if w != 0 and h != 0:
            img = img.resize((w, h), Image.BILINEAR)
    out.append(np.array(img.convert("RGB")))


imgout = np.array(out)
imgout = imgout.transpose(*[outdict[c] for c in outfmt]) / 255.0

print("got packed", imgout.shape, imgout.dtype)

scipy.io.savemat(args.output, {"images": imgout})

print("saved mat shape", imgout.shape, "to", args.output)
