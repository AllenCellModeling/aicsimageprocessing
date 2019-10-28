#!/usr/bin/env python

# authors: Dan Toloudis danielt@alleninstitute.org
#          Zach Crabtree zacharyc@alleninstitute.org

import oldaicsimageio

import math
import numpy as np
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw
import platform
import skimage.transform
from typing import List, Tuple, Sequence, Union

COLORS_CMY = ((0.0, 1.0, 1.0), (1.0, 0.0, 1.0), (1.0, 1.0, 0.0))
DEFAULT_SIZE = 128


def resize_cyx_image(image, new_size):
    """
    This function resizes a CYX image.

    :param image: CYX ndarray
    :param new_size: tuple of shape of desired image dimensions in CYX
    :return: image with shape of new_size
    """
    scaling = float(image.shape[1]) / float(new_size[1])
    # get the shape of the image that is resized by the scaling factor
    test_shape = np.ceil(np.divide(image.shape, [1, scaling, scaling]))
    # sometimes the scaling can be rounded incorrectly and scale the image to
    # one pixel too high or too low
    if not np.array_equal(test_shape, new_size):
        # getting the scaling from the other dimension solves this rounding problem
        scaling = float(image.shape[2]) / float(new_size[2])
        test_shape = np.ceil(np.divide(image.shape, [1, scaling, scaling]))
        # if neither scaling factors achieve the desired shape, then the aspect ratio of the image
        # is different than the aspect ratio of new_size
        if not np.array_equal(test_shape, new_size):
            raise ValueError("This image does not have the same aspect ratio as new_size")

    image = image.transpose((2, 1, 0))

    if scaling < 1:
        scaling = 1.0 / scaling
        im_out = skimage.transform.pyramid_expand(image, upscale=scaling, multichannel=True)
    elif scaling > 1:
        im_out = skimage.transform.pyramid_reduce(image, downscale=scaling, multichannel=True)
    else:
        im_out = image

    im_out = im_out.transpose((2, 1, 0))
    assert im_out.shape == new_size

    return im_out


def create_projection(image: np.ndarray, axis: int, method: str = 'max', slice_index: int = 0, sections: int = 3):
    """
    This function creates a 2D projection out of an n-dimensional image.

    :param image: ZCYX array
    :param axis: the axis that the projection should be performed along
    :param method: the method that will be used to create the projection
                   Options: ["max", "mean", "sum", "slice", "sections"]
                   - max will look through each axis-slice, and determine the max value for each pixel
                   - mean will look through each axis-slice, and determine the mean value for each pixel
                   - sum will look through each axis-slice, and sum all pixels together
                   - slice will take the pixel values from the middle slice of the stack
                   - sections will split the stack into `sections` number of sections, and take a
                   max projection for each.
    :param slice_index: index to use for the 'slice' projection method
    :param sections: number of sections to select and max-intensity project for the 'sections' projection method
    :return:
    """
    if method == 'max':
        image = np.max(image, axis)
    elif method == 'mean':
        image = np.mean(image, axis)
    elif method == 'sum':
        image = np.sum(image, axis)
    elif method == 'slice':
        image = image[slice_index]
    elif method == 'sections':
        separator = int(math.floor(image.shape[0] / sections))
        # stack is a 2D YX im
        stack = np.zeros(image[0].shape)
        for i in range(sections - 1):
            bottom_bound = separator * i
            top_bound = separator + bottom_bound
            # TODO: this line assumes the stack is separated through the z-axis, instead of the designated axis param
            section = np.max(image[bottom_bound:top_bound], axis)
            stack += section
        stack += np.max(image[separator * sections - 1:])

        return stack
    # returns 2D image, YX
    return image


def subtract_noise_floor(image, bins=256):
    # image is a 3D ZYX image
    immin = image.min()
    immax = image.max()
    hi, bin_edges = np.histogram(image, bins=bins, range=(immin, immax))
    # index of tallest peak in histogram
    peakind = np.argmax(hi)
    # subtract this out
    subtracted = image.astype(np.float32)
    subtracted -= bin_edges[peakind]
    # don't go negative
    subtracted[subtracted < 0] = 0
    return subtracted


# assumes sizes are c, x, y so the 1 and 2 indices are the x and y
def _get_letterbox_bounds(full_size, scaled_size):
    x0 = (full_size[1] - scaled_size[1]) // 2
    x1 = x0 + scaled_size[1]
    y0 = (full_size[2] - scaled_size[2]) // 2
    y1 = y0 + scaled_size[2]
    return x0, x1, y0, y1


class ThumbnailGenerator:
    """

    This class is used to generate thumbnails for 4D CZYX images.

    Example:
        generator = ThumbnailGenerator()
        for image in image_array:
            thumbnail = generator.make_thumbnail(image)

    """

    def __init__(self, colors: Sequence[Tuple[float, float, float]] = COLORS_CMY, size: int = DEFAULT_SIZE,
                 channel_indices: Sequence[int] = None, channel_thresholds: Sequence[float] = None,
                 channel_multipliers: Sequence[int] = None, mask_channel_index: int = 5, letterbox: bool = True,
                 projection: str = 'max', projection_sections: int = 3, return_rgb: bool = True):
        """
        :param colors: The color palette that will be used to color each channel. The default palette
                       colors are magenta=membrane, nucleus=cyan, structure=yellow.
                       Keep color-blind accessibility in mind.

        :param size: This constrains the image to have the X or Y dims max out at this value, but keep
                     the original aspect ratio of the image.

        :param channel_indices: An array of channel indices to represent the three main channels of the cell

        :param mask_channel_index: The index for the segmentation channel in image that will be used to mask
                                   the thumbnail

        :param projection: The method that will be used to generate each channel's projection. This is done
                           for each pixel, through the z-axis
                           Options: ["max", "slice", "sections"]
                           - max will look through each z-slice, and determine the max value for each pixel
                           - slice will take the pixel values from the middle slice of the z-stack
                           - sections will split the zstack into projection_sections number of sections, and take a
                             max projection for each.

        :param projection_sections: The number of sections that will be used to determine projections,
                                    if projection="sections"

        :param return_rgb: Return an array that has been clipped and cast as uint8 to make it RGB compatible.
                           Setting this to False will return the float array that is (generally) bounded between
                           0 and 1.
        """

        if channel_indices is None:
            channel_indices = [0, 1, 2]
        if channel_thresholds is None:
            channel_thresholds = [.65, .65, .65]
        if channel_multipliers is None:
            channel_multipliers = [1, 1, 1]

        self.colors = colors
        self.size = size
        self.channel_indices = channel_indices
        self.channel_thresholds = channel_thresholds
        self.channel_multipliers = channel_multipliers
        self.mask_channel_index = mask_channel_index
        self.letterbox = letterbox
        self.projection = projection
        self.projection_sections = projection_sections
        self.return_rgb = return_rgb

        self._validate_settings()

    def _validate_settings(self):
        assert self.projection in ["slice", "max",  "sections"]

        assert len(self.colors) > 0 and all(len(color) == 3 for color in self.colors), \
            f"Colors {self.colors} are invalid"

        assert len(self.colors) == len(self.channel_indices), (
            f"Colors palette is a different size than the channel indices "
            f"(len({self.colors}) != len({self.channel_indices}))"
        )
        assert min(self.channel_indices) >= 0, "Minimum channel index must be greater than or equal to 0"

        assert len(self.channel_thresholds) >= len(self.channel_indices)

        assert len(self.channel_multipliers) >= len(self.channel_indices)

    def _make_thumbnail(self, image: np.ndarray, new_size: Union[Tuple[int, int, int], np.ndarray],
                        output_size_dim: Tuple[int, int, int], apply_cell_mask: bool = False) -> np.ndarray:
        if apply_cell_mask:
            shape_out_rgb = new_size

            # apply the cell segmentation mask.  bye bye to data outside the cell
            # for i in range(len(self.channel_indices)):
            #     image[:, i] = np.multiply(image[:, i], image[:, self.mask_channel_index] > 0)

            num_noise_floor_bins = 32
            composite = np.zeros((shape_out_rgb[0], output_size_dim[1], output_size_dim[2]))
            for i in range(len(self.channel_indices)):
                ch = self.channel_indices[i]
                # try to subtract out the noise floor.
                # range is chosen to ignore zeros due to masking.  alternative is to pass mask image as weights=im1[-1]
                thumb = subtract_noise_floor(image[:, ch], bins=num_noise_floor_bins)
                # apply mask
                thumb = np.multiply(thumb, image[:, self.mask_channel_index] > 0)

                # renormalize
                thmax = thumb.max()
                thumb /= thmax

                # resize before projection?
                rgbproj = np.asarray(thumb)
                rgbproj = create_projection(rgbproj, 0, self.projection, slice_index=rgbproj.shape[0] // 2)
                rgb_out = np.expand_dims(rgbproj, 2)
                rgb_out = np.repeat(rgb_out, 3, 2)

                # inject color.  careful of type mismatches.
                rgb_out *= self.colors[i]

                rgb_out /= np.max(rgb_out)

                rgb_out = resize_cyx_image(rgb_out.transpose((2, 1, 0)), shape_out_rgb).astype(np.float32)

                x0, x1, y0, y1 = _get_letterbox_bounds(output_size_dim, shape_out_rgb)
                composite[:, x0:x1, y0:y1] += rgb_out
            # renormalize
            composite /= composite.max()
            # return as cyx for pngwriter
            return composite.transpose((0, 2, 1))
        else:
            image = image.transpose((1, 0, 2, 3))
            shape_out_rgb = new_size

            num_noise_floor_bins = 16
            composite = np.zeros((shape_out_rgb[0], output_size_dim[1], output_size_dim[2]))
            channel_indices = self.channel_indices
            rgb_image = []
            for i in channel_indices:
                # subtract out the noise floor.
                immin = image[i].min()
                immax = image[i].max()
                hi, bin_edges = np.histogram(image[i], bins=num_noise_floor_bins, range=(max(1, immin), immax))
                # index of tallest peak in histogram
                peakind = np.argmax(hi)
                # subtract this out
                thumb = image[i].astype(np.float32)
                thumb -= bin_edges[peakind]
                # don't go negative
                thumb[thumb < 0] = 0
                # renormalize
                thmax = thumb.max()
                thumb /= thmax

                imdbl = np.asarray(thumb).astype('double')
                im_proj = create_projection(imdbl, 0, self.projection, slice_index=int(thumb.shape[0] // 2))

                # Add the modified channel to the list of channels to composite
                rgb_image.append(im_proj)

            # Composite the desired channels
            # rgb_image and self.colors can safely be assumed the same length because
            # rgb_image is the same length as self.channel_indices and an assertion
            # is made to that effect when constructing the class
            for channel, color in zip(rgb_image, self.colors):
                # turn into RGB
                rgb_out = np.expand_dims(channel, 2)
                rgb_out = np.repeat(rgb_out, 3, 2).astype('float')

                # inject color.  careful of type mismatches.
                rgb_out *= color

                rgb_out /= np.max(rgb_out)

                rgb_out = resize_cyx_image(rgb_out.transpose((2, 1, 0)), shape_out_rgb)

                x0, x1, y0, y1 = _get_letterbox_bounds(output_size_dim, shape_out_rgb)
                composite[:, x0:x1, y0:y1] += rgb_out

            # returns a CYX array for the pngwriter
            return composite.transpose((0, 2, 1))

    def _get_output_shape(self, im_size: Union[Tuple[int, int, int], np.ndarray]) -> Tuple[int, int, int]:
        """
        This method will take in a 3D ZYX shape and return a 3D XYC of the final thumbnail

        :param im_size: 3D ZYX shape of original image
        :return: CYX dims for a resized thumbnail where the maximum X or Y dimension is the one
                 specified in the constructor.
        """
        # size down to this edge size, maintaining aspect ratio.
        max_edge = self.size
        # keep same number of z slices.
        shape_out = np.hstack((im_size[0],
                               max_edge if im_size[1] > im_size[2] else max_edge * (float(im_size[1]) / im_size[2]),
                               max_edge if im_size[1] < im_size[2] else max_edge * (float(im_size[2]) / im_size[1])
                               ))
        return 3, int(np.ceil(shape_out[2])), int(np.ceil(shape_out[1]))

    def make_thumbnail(self, image: np.ndarray, apply_cell_mask: bool = False) -> np.ndarray:
        """
        This method is the primary interface with the ThumbnailGenerator. It can be used many times with different
        images in order to save the configuration that was specified at the beginning of the generator.

        :param image: ZCYX image that is the source of the thumbnail
        :param apply_cell_mask: boolean value that designates whether the image is a fullfield or segmented cell
                                False -> fullfield, True -> segmented cell
        :return: a CYX image, scaled down to the size designated in the constructor
        """

        image = image.astype(np.float32)
        # check to make sure there are 3 or more channels
        assert image.shape[1] >= 3, "The image did not have 3 or more channels"
        assert image.shape[2] > 1 and image.shape[3] > 1
        assert max(self.channel_indices) <= image.shape[1] - 1
        if apply_cell_mask:
            assert self.mask_channel_index <= image.shape[1]

        im_size = np.array(image[:, 0].shape)
        assert len(im_size) == 3
        shape_out_rgb = self._get_output_shape(im_size)

        final_size = (shape_out_rgb[0], self.size, self.size) if self.letterbox else shape_out_rgb

        thumbnail = self._make_thumbnail(image, shape_out_rgb, final_size, apply_cell_mask=apply_cell_mask)

        if not self.return_rgb:
            return thumbnail

        # Clip the values of the float array between 0 and 1, rescale to the bit depth of uint8,
        # and cast to uint8
        rgb_thumbnail = (thumbnail.clip(0, 1) * 255.).astype(np.uint8)

        return rgb_thumbnail


def make_one_thumbnail(infile: str,
                       outfile: str,
                       channels: List[int],
                       colors: List[Tuple[float, float, float]],
                       size: int,
                       projection: str = 'max',
                       axis: int = 2,
                       apply_mask: bool = False,
                       mask_channel: int = 0,
                       label: str = ''):
    axistranspose = (1, 0, 2, 3)
    if axis == 2:  # Z
        axistranspose = (1, 0, 2, 3)
    elif axis == 0:  # X
        axistranspose = (2, 0, 1, 3)
    elif axis == 1:  # Y
        axistranspose = (3, 0, 2, 1)
    else:
        raise ValueError(f'Unknown axis value: {axis}')

    image = oldaicsimageio.AICSImage(infile)
    imagedata = image.get_image_data()
    generator = ThumbnailGenerator(channel_indices=channels,
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
        thumbnail = np.array(img)
        thumbnail = thumbnail.transpose(2, 0, 1)

    with oldaicsimageio.PngWriter(file_path=outfile, overwrite_file=True) as writer:
        writer.save(thumbnail)
    return thumbnail
