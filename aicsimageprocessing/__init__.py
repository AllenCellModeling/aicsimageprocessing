from .version import MODULE_VERSION

from .alignMajor import *
from .backgroundSub import *
from .backgroundCrop import *
from .flip import *
from .imgCenter import *
from .imgToProjection import *
from .isosurfaceGenerator import *
from .resize import *
from .thumbnailGenerator import *


def get_version():
    return MODULE_VERSION
