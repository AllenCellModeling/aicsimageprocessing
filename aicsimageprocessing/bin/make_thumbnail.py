import argparse
import logging
import sys
import traceback

from aicsimageprocessing import get_module_version, thumbnailGenerator

###############################################################################

log = logging.getLogger()
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s - %(name)s - %(lineno)3d][%(levelname)s] %(message)s",
)

###############################################################################


class Args(argparse.Namespace):
    def __init__(self):
        # Arguments that could be passed in through the command line
        self.debug = False
        #
        self.__parse()

    def __parse(self):
        p = argparse.ArgumentParser(
            prog="make_thumbnail", description="Make a thumbnail from a ome-tiff"
        )
        p.add_argument(
            "-v",
            "--version",
            action="version",
            version="%(prog)s " + get_module_version(),
        )

        p.add_argument(
            "--debug", action="store_true", dest="debug", help=argparse.SUPPRESS
        )

        p.add_argument("infile", type=str, help="input zstack")
        p.add_argument("outfile", type=str, help="output png")

        # assume square for now
        p.add_argument("--size", type=int, help="size", default=128)
        p.add_argument("--mask", type=int, help="mask channel", default=-1)
        p.add_argument("--axis", type=int, help="axis 0, 1, or 2", default=2)
        p.add_argument(
            "--channels", type=int, nargs="+", help="channels to composite", default=[0]
        )
        p.add_argument(
            "--colors",
            type=str,
            nargs="+",
            help="colors to composite, one per channel",
            default=["ffffff"],
        )
        p.add_argument(
            "--projection", type=str, help="projection type max or slice", default="max"
        )
        p.add_argument("--label", type=str, help="string label on image", default="")

        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()
        dbg = args.debug
        colors = [
            (tuple(int(h[i : i + 2], 16) / 255.0 for i in (0, 2, 4)))
            for h in args.colors
        ]
        thumbnailGenerator.make_one_thumbnail(
            infile=args.infile,
            outfile=args.outfile,
            channels=args.channels,
            colors=colors,
            size=args.size,
            projection=args.projection,
            axis=args.axis,
            apply_mask=(args.mask != -1),
            mask_channel=args.mask,
        )
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
