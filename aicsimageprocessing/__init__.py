# -*- coding: utf-8 -*-

"""Top-level package for aicsimageprocessing."""

__author__ = "AICS"
__email__ = "!AICS_SW@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.7.4"


def get_module_version():
    return __version__


from .alignMajor import *  # noqa
from .backgroundCrop import *  # noqa
from .backgroundSub import *  # noqa
from .crop_img import *  # noqa
from .flip import *  # noqa
from .imgCenter import *  # noqa
from .imgToCoords import *  # noqa
from .imgToProjection import *  # noqa
from .imshow import *  # noqa
from .isosurfaceGenerator import *  # noqa
from .normalization import *  # noqa
from .resize import *  # noqa
from .rigidAlignment import *  # noqa
from .thumbnailGenerator import *  # noqa
