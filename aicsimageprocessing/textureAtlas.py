# author: Zach Crabtree zacharyc@alleninstitute.org

import json
import math as m
import os

import numpy as np
import skimage.transform as sktransform
from aicsimageio import AICSImage
from aicsimageio.writers.two_d_writer import TwoDWriter
from aicsimageio.types import PhysicalPixelSizes


class TextureAtlasDims:
    def __init__(self):
        self.width = 0
        self.height = 0
        self.channels = 0
        self.channel_names = []
        self.rows = 0
        self.cols = 0
        self.tiles = 0
        self.tile_width = 0
        self.tile_height = 0
        self.atlas_width = 0
        self.atlas_height = 0
        self.pixel_size_x = 1
        self.pixel_size_y = 1
        self.pixel_size_z = 1


class TextureAtlas:
    def __init__(self, aics_image, pack_order, dims, t=0):
        if not isinstance(dims, TextureAtlasDims):
            raise ValueError(
                "Texture atlas dimension data must be of type TextureAtlasDims!"
            )
        if not isinstance(aics_image, AICSImage):
            raise ValueError(
                "Texture atlases can only be generated with AICSImage objects!"
            )
        self.aics_image = aics_image

        if len(pack_order) > 4:
            raise ValueError(
                "An atlas with more than 4 channels ({}) cannot be created!".format(
                    pack_order
                )
            )
        if any(channel > self.aics_image.dims.C for channel in pack_order):
            raise IndexError(
                (
                    "A channel specified in the ordering {} is out-of-bounds in"
                    " the AICSImage object!"
                ).format(pack_order)
            )

        self.pack_order = pack_order
        self.metadata = {"name": "NOT_YET_ASSIGNED", "channels": self.pack_order}
        self.atlas = self.generate_atlas(dims, t)

    def generate_atlas(self, dims, t):
        atlas = np.stack(
            [
                self._atlas_single_channel(image_channel, dims, t)
                for image_channel in self.pack_order
            ]
        )
        return atlas

    def _atlas_single_channel(self, channel, dims, t):
        scale = (
            float(dims.tile_width) / float(self.aics_image.dims.X),
            float(dims.tile_height) / float(self.aics_image.dims.Y),
        )

        channel_data = self.aics_image.get_image_data("XYZ", C=channel, T=t)
        # renormalize
        channel_data = channel_data.astype(np.float32)

        atlas = np.zeros((dims.atlas_width, dims.atlas_height))
        i = 0
        for row in range(dims.rows):
            top_bound, bottom_bound = (
                (dims.tile_height * row),
                (dims.tile_height * (row + 1)),
            )
            for col in range(dims.cols):
                if i < self.aics_image.dims.Z:
                    left_bound, right_bound = (
                        (dims.tile_width * col),
                        (dims.tile_width * (col + 1)),
                    )
                    tile = sktransform.rescale(
                        channel_data[:, :, i], scale, preserve_range=True
                    )
                    atlas[left_bound:right_bound, top_bound:bottom_bound] = tile
                    i += 1
                else:
                    break

        # rescale min to 0 and max to 255
        mn = min(0, atlas.min())
        mx = atlas.max()
        atlas = 255.0 * (atlas - mn) / (mx - mn)
        # atlas = np.interp(atlas, (min(0, atlas.min()), atlas.max()), (0.0, 255.0))
        atlas = atlas.astype(np.uint8)
        return atlas


class TextureAtlasGroup:
    def __init__(
        self, aics_image, t=0, name="texture_atlas", pack_order=None, max_edge=2048
    ):
        self.name = name
        self.max_edge = max_edge
        self.stack_height = aics_image.dims.Z
        self.dims = self._calc_atlas_dimensions(aics_image)
        self.atlas_list = []

        max_channels_per_png = 3
        if pack_order is None:
            # if no pack order is specified, pack channels into png and move on
            channel_list = list(range(aics_image.dims.C))
            pack_order = [
                channel_list[x : x + max_channels_per_png]
                for x in range(0, len(channel_list), max_channels_per_png)
            ]
        png_count = 0
        for png in pack_order:
            self._append(TextureAtlas(aics_image, pack_order=png, dims=self.dims, t=t))
            png_count += 1

    def _calc_atlas_dimensions(self, aics_image):
        tile_width, tile_height, stack_height = (
            aics_image.dims.X,
            aics_image.dims.Y,
            aics_image.dims.Z,
        )
        # maintain aspect ratio of images
        # initialize atlas with one row of all slices
        atlas_width = tile_width * stack_height
        atlas_height = tile_height
        ratio = float(atlas_width) / float(atlas_height)
        # these next steps attempt to optimize the atlas into a square shape
        # TODO - there must be a way to do this with a single calculation
        for r in range(2, stack_height):
            new_rows = m.ceil(float(stack_height) / r)
            adjusted_width = int(tile_width * new_rows)
            adjusted_height = int(tile_height * r)
            new_ratio = float(max(adjusted_width, adjusted_height)) / float(
                min(adjusted_width, adjusted_height)
            )
            if new_ratio < ratio:
                ratio = new_ratio
                atlas_width = adjusted_width
                atlas_height = adjusted_height
            else:
                # we've found the rows and columns that make this the most square image
                break
        cols = int(atlas_width // tile_width)
        rows = int(atlas_height // tile_height)

        if self.max_edge < atlas_width or self.max_edge < atlas_height:
            tile_width = m.floor(self.max_edge / cols)
            tile_height = m.floor(self.max_edge / rows)
            atlas_width = tile_width * cols
            atlas_height = tile_height * rows

        dims = TextureAtlasDims()
        dims.tile_width = int(tile_width)
        dims.tile_height = int(tile_height)
        dims.rows = int(rows)
        dims.cols = int(cols)
        dims.atlas_width = int(atlas_width)
        dims.atlas_height = int(atlas_height)
        dims.width = aics_image.dims.X
        dims.height = aics_image.dims.Y
        dims.channels = aics_image.dims.C
        dims.tiles = aics_image.dims.Z

        channel_names = aics_image.channel_names
        if channel_names is not None:
            dims.channel_names = channel_names
        else:
            dims.channel_names = ["CH_" + str(i) for i in range(aics_image.dims.C)]

        try:
            physical_pixel_size = aics_image.physical_pixel_sizes
        except AttributeError:
            physical_pixel_size = PhysicalPixelSizes(1.0, 1.0, 1.0)
        if physical_pixel_size is not None:
            dims.pixel_size_x = physical_pixel_size.X
            dims.pixel_size_y = physical_pixel_size.Y
            dims.pixel_size_z = physical_pixel_size.Z
        else:
            dims.pixel_size_x = 1
            dims.pixel_size_y = 1
            dims.pixel_size_z = 1

        return dims

    def _is_valid_atlas(self, atlas):
        if atlas is None:
            return False
        if atlas.atlas is None:
            return False
        if atlas.atlas.shape is None:
            return False
        shape = atlas.atlas.shape
        if self.dims.atlas_width != shape[1]:
            print(
                f"width mismatch atlas_width={self.dims.atlas_width}"
                f" but shape is {shape[1]}"
            )
            return False
        if self.dims.atlas_height != shape[2]:
            print(
                f"height mismatch atlas_height={self.dims.atlas_height}"
                f" but shape is {shape[2]}"
            )
            return False
        return True

    def _append(self, atlas):
        if not isinstance(atlas, TextureAtlas):
            raise ValueError("TextureAtlasGroup can only append TextureAtlas objects!")
        if self._is_valid_atlas(atlas):
            self.atlas_list.append(atlas)
        else:
            raise ValueError(
                "Attempted to add atlas that doesn't match the rest of atlasGroup"
            )

    def get_metadata(self):
        metadata = self.dims.__dict__
        metadata["images"] = [atlas.metadata for atlas in self.atlas_list]
        metadata["name"] = self.name
        return metadata

    def save(self, output_dir, name=None, user_data=None):
        """
        Saves a TextureAtlasGroup into one json file and 1 to many png files. Files are
        named with formats `name+'_atlas.json'` and `name+'_atlas_N.png'`

        Parameters
        ----------
        output_dir
            directory in which to write the files

        name
            if not supplied, then use the name given when creating the TextureAtlasGroup

        user_data
            optional dictionary of additional data to add to json file; to be used by
            client application
        """
        if name is None:
            name = self.name
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        i = 0
        for atlas in self.atlas_list:
            atlasname = name + "_atlas_" + str(i) + ".png"
            # add this name to the atlas's metadata for use in get_metadata below.
            atlas.metadata["name"] = atlasname
            full_path = os.path.join(output_dir, atlasname)
            TwoDWriter.save(atlas.atlas, full_path, dim_order="SXY")
            i += 1

        metadata = self.get_metadata()
        if user_data is not None:
            metadata["userData"] = user_data
        with open(os.path.join(output_dir, name + "_atlas.json"), "w") as json_output:
            json.dump(metadata, json_output)


def generate_texture_atlas(
    im, name="texture_atlas", max_edge=2048, pack_order=None, t=0
):
    """
    Creates a TextureAtlasGroup object

    Parameters
    ----------
    im
        aicsImage object

    name
        will be stored in metadata and used at save time if no name is supplied at that
        time

    max_edge
        this designates the largest side in the texture atlas

    pack_order
        a 2d list that contains what channel in the image should be saved to the RGBA
        values in the final png. for example, a 7 channel image might be saved like
        [[0, 1, 2, 3], [4, 5], [6]] where the first texture atlas will code channel 0
        as r, channel 1 as g, and so on.

    t
        time index, zero by default.

    Returns
    -------
    TextureAtlasGroup object
    """
    atlas_group = TextureAtlasGroup(
        im, name=name, max_edge=max_edge, pack_order=pack_order, t=t
    )
    return atlas_group
