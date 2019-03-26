#!/usr/bin/env python

# author: Zach Crabtree zacharyc@alleninstitute.org

import numpy as np
import pytest

from aicsimageprocessing.thumbnailGenerator import ThumbnailGenerator


"""Constructor tests"""


@pytest.mark.parametrize(
    'color_palette', [
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        pytest.param([['a', 'b', 'c', 'd']], marks=pytest.mark.raises(exception=AssertionError,
                                                                      match="Colors .*? are invalid"))
    ]
)
def test_colors_constructor(color_palette):
    # act, assert
    generator = ThumbnailGenerator(colors=color_palette)
    # Assert that there are rgb values for each channel in channel_indices.
    assert generator is not None


@pytest.mark.parametrize(
    'channel_indices', [
        [0, 1, 2],

        # Channel indices is not the same size as color palette
        pytest.param([0, 1, 2, 3], marks=pytest.mark.raises(exception=AssertionError,
                                                            match="Colors palette is a different size .*")),

        # Minimum channel index must be greater than 0
        pytest.param([-1, -1, -1],
                     marks=pytest.mark.raises(exception=AssertionError,
                                              match="Minimum channel index must be greater than 0"))
    ]
)
def test_channel_indices_constructor(channel_indices):
    # act
    generator = ThumbnailGenerator(channel_indices=channel_indices)
    # assert
    assert generator is not None

"""make_thumbnail tests"""

def test_MakeValidThumbnail(self):
    # arrange
    valid_image = np.random.rand(10, 7, 256, 256)
    generator = ThumbnailGenerator(size=128)

    # act
    valid_thumbnail = generator.make_thumbnail(valid_image)

    # assert
    self.assertEqual(valid_thumbnail.shape, (4, 128, 128),
                     msg="The thumbnail was rescaled to rgba channels and "
                         "height/width with same aspect ratio of initial image")

def test_MakeInvalidThumbnail(self):
    # arrange
    invalid_image = np.random.rand(1, 2, 128, 128)  # < 3 channels should be an invalid image
    generator = ThumbnailGenerator(size=128)

    # act, assert
    with self.assertRaises(Exception, msg="The image did not have more than 2 channels"):
        generator.make_thumbnail(invalid_image)
