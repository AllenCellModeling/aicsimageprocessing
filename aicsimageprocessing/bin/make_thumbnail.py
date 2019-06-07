import aicsimageio

from aicsimageprocessing import get_module_version, thumbnailGenerator

import argparse
import logging
import numpy
import platform
import sys
import traceback
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw

###############################################################################

log = logging.getLogger()
# Note: basicConfig should only be called in bin scripts (CLIs).
# https://docs.python.org/3/library/logging.html#logging.basicConfig
# "This function does nothing if the root logger already has handlers configured for it."
# As such, it should only be called once, and at the highest level (the CLIs in this case).
# It should NEVER be called in library code!
logging.basicConfig(level=logging.INFO,
                    format='[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s')

###############################################################################


class Args(argparse.Namespace):

    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(prog='make_thumbnail', description='Make a thumbnail from a ome-tiff')
        p.add_argument('-v', '--version', action='version', version='%(prog)s ' + get_module_version())
        p.add_argument('--debug', action='store_true', dest='debug', help=argparse.SUPPRESS)

        p.add_argument('infile', type=str, help='input zstack')
        p.add_argument('outfile', type=str, help='output png')

        # assume square for now
        p.add_argument('--size', type=int, help='size', default=128)
        p.add_argument('--mask', type=int, help='mask channel', default=-1)
        p.add_argument('--axis', type=int, help='axis 0, 1, or 2', default=2)
        p.add_argument('--channels', type=int, nargs='+', help='channels to composite', default=[0])
        p.add_argument('--colors', type=str, nargs='+', help='colors to composite, one per channel', default=['ffffff'])
        p.add_argument('--projection', type=str, help='projection type max or slice', default='max')
        p.add_argument('--label', type=str, help='string label on image', default='')

        p.parse_args(namespace=self)


###############################################################################

def make_one_thumbnail(infile, outfile, channels, colors, size, projection='max', axis=2, apply_mask=False, mask_channel=0, label=''):
    axistranspose = (1, 0, 2, 3)
    if axis == 2:  # Z
        axistranspose = (1, 0, 2, 3)
    elif axis == 0:  # X
        axistranspose = (2, 0, 1, 3)
    elif axis == 1:  # Y
        axistranspose = (3, 0, 2, 1)
    else:
        raise ValueError(f'Unknown axis value: {axis}')

    image = aicsimageio.AICSImage(infile)
    imagedata = image.get_image_data()
    generator = thumbnailGenerator.ThumbnailGenerator(channel_indices=channels,
                                                      size=size,
                                                      mask_channel_index=mask_channel,
                                                      colors=colors,
                                                      projection=projection)
    # take zeroth time, and transpose projection axis and c
    thumbnail = generator.make_thumbnail(imagedata[0].transpose(axistranspose), apply_cell_mask=apply_mask)
    if label:
        # Untested on MacOS
        if platform.system() == "Windows":
            font_path = "/Windows/Fonts/consola.ttf"
        else:
            font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
        font = ImageFont.truetype(font_path, 12)
        img = Image.fromarray(thumbnail.transpose((1, 2, 0)))
        draw = ImageDraw.Draw(img)
        draw.text((2, 2), label, (255, 255, 255), font=font)
        thumbnail = numpy.array(img)
        thumbnail = thumbnail.transpose(2, 0, 1)

    with aicsimageio.PngWriter(file_path=outfile, overwrite_file=True) as writer:
        writer.save(thumbnail)
    return thumbnail


def main():
    try:
        args = Args()
        dbg = args.debug
        colors = [(tuple(int(h[i:i+2], 16)/255.0 for i in (0, 2, 4))) for h in args.colors]
        make_one_thumbnail(
            infile=args.infile,
            outfile=args.outfile,
            channels=args.channels,
            colors=colors,
            size=args.size,
            projection=args.projection,
            axis=args.axis,
            apply_mask=(args.mask != -1),
            mask_channel=args.mask)
    except Exception as e:
        log.error("=============================================")
        if dbg:
            log.error("\n\n" + traceback.format_exc())
            log.error("=============================================")
        log.error("\n\n" + str(e) + "\n")
        log.error("=============================================")
        sys.exit(1)


if __name__ == "__main__":
    main()

