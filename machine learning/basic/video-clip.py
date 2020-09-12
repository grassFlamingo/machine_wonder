import argparse
import ffmpeg
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("-i", "--input", type=str, help="input file")
parser.add_argument("-s", "--size", type=int,
                    help="size of crop rect", nargs=2, default=(640, 360))
parser.add_argument("-c", "--center", type=int,
                    help="center of crop rect; default is the center of orignal video.", nargs=2, default=None)
parser.add_argument("-o", "--output", type=str,
                    help="output file", default="video-out.mp4")

args = parser.parse_args()


def get_resolution(infile):
    """
    return width, height, num_frames
    """
    try:
        probe = ffmpeg.probe(infile)
    except ffmpeg.Error as e:
        print(e.stderr, file=sys.stderr)
        return 0, 0, 0

    video_stream = next(
        (stream for stream in probe['streams'] if stream['codec_type'] == 'video'), None)
    if video_stream is None:
        print('No video stream found', file=sys.stderr)
        return 0, 0, 0

    width = int(video_stream['width'])
    height = int(video_stream['height'])
    num_frames = int(video_stream['nb_frames'])

    return width, height, num_frames


def crop_video(infile, outfile, size, center=None):
    """
    crop image using rect
    if position is None:
        the center of rect is the center of each frame
    else:
        the center of rect is `position`

    Requires:
    - infile: input video file name
    - outfile: output video file name
    - size [tuple|list]: tuple or list of length 2; 
        size[0] is the width of crop rect
        size[1] is the height of crop rect
    - center: center of rect
    """
    w, h, _ = get_resolution(infile)
    if center is None:
        center = (w//2, h//2)
    else:
        center = tuple(center)
    size = tuple(size)
    # top left
    postl = (center[0] - size[0]//2, center[1] - size[1]//2)
    if postl[0] < 0 or postl[1] < 0:
        print("rect {} center at {} is out of image!\n".format(size, center))
        return

    print("convering video {} {} to video {} {}; center at {}".format(
        infile, (w, h), outfile, size, center))
    instream = ffmpeg.input(infile)
    video = ffmpeg.crop(instream.video, postl[0], postl[1], size[0], size[1])
    outstream = ffmpeg.output(video, instream.audio, outfile)
    ffmpeg.run(outstream, quiet=True, overwrite_output=True)
    print("conver succeed")


crop_video(args.input, args.output, args.size, args.center)

