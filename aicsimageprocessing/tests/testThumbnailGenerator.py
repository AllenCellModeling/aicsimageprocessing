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
                                              match="Minimum channel index must be greater than or equal to 0"))
    ]
)
def test_channel_indices_constructor(channel_indices):
    # act
    generator = ThumbnailGenerator(channel_indices=channel_indices)
    # assert
    assert generator is not None


"""make_thumbnail tests"""


@pytest.mark.parametrize('thumbnail_size', [128, 256])
@pytest.mark.parametrize('image_shape', [
    (10, 7, 256, 256),
    (10, 7, 128, 128),
    (1, 4, 256, 256),
    (1, 4, 64, 64),
    pytest.param((1, 2, 128, 128),
                 marks=pytest.mark.raises(exception=Exception,
                                          match="The image did not have 3 or more channels"))
])
@pytest.mark.parametrize('return_rgb', [True, False])
def test_thumbnail_generation(thumbnail_size, image_shape, return_rgb):
    image = np.random.randint(low=1, high=2000, size=image_shape)

    # arrange
    generator = ThumbnailGenerator(size=thumbnail_size, projection='slice', return_rgb=return_rgb)

    # act
    thumbnail = generator.make_thumbnail(image)

    assert thumbnail.shape == (3, thumbnail_size, thumbnail_size)

    if return_rgb:
        assert thumbnail.dtype == np.uint8
    else:
        assert thumbnail.dtype == np.float64
