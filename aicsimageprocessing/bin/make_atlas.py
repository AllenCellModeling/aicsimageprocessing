import argparse
import logging
import os
import sys
import traceback

from aicsimageprocessing import get_module_version, textureAtlas
from aicsimageio import AICSImage

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
            prog="make_atlas",
            description="Make a volume-viewer texture atlas from a ome-tiff",
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
        p.add_argument("outdir", type=str, help="output directory")

        p.parse_args(namespace=self)


###############################################################################


def main():
    try:
        args = Args()

        dbg = args.debug
        image = AICSImage(args.infile)
        # preload for performance
        image.data
        name = os.path.splitext(os.path.basename(args.infile))[0]
        atlas_group = textureAtlas.generate_texture_atlas(
            image, name=name, max_edge=2048, pack_order=None
        )
        atlas_group.save(args.outdir)
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
