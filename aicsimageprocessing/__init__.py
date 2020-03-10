# -*- coding: utf-8 -*-

"""Top-level package for aicsimageprocessing."""

__author__ = "AICS"
__email__ = "!AICS_SW@alleninstitute.org"
# Do not edit this string manually, always use bumpversion
# Details in CONTRIBUTING.md
__version__ = "0.7.2"


def get_module_version():
    return __version__


from .alignMajor import *  # noqa: E402
from .backgroundCrop import *  # noqa: E402
from .backgroundSub import *  # noqa: E402
from .crop_img import *  # noqa: E402
from .flip import *  # noqa: E402
from .imgCenter import *  # noqa: E402
from .imgToCoords import *  # noqa: E402
from .imgToProjection import *  # noqa: E402
from .imshow import *  # noqa: E402
from .isosurfaceGenerator import *  # noqa: E402
from .normalization import *  # noqa: E402
from .resize import *  # noqa: E402
from .rigidAlignment import *  # noqa: E402
from .thumbnailGenerator import *  # noqa: E402
