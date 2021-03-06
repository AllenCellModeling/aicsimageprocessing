# author: Zach Crabtree zacharyc@alleninstitute.org

import unittest

from aicsimageio import AICSImage
from aicsimageprocessing.textureAtlas import generate_texture_atlas


class TestTextureAtlas(unittest.TestCase):
    @unittest.skip("temporarily disabled")
    def test_Save(self):
        with AICSImage("img/img40_1.ome.tif") as image:
            atlas = generate_texture_atlas(
                im=image,
                pack_order=[[0, 1, 2], [3], [4]],
                prefix="test_Sizing",
                max_edge=1024,
            )
        atlas.save("img/atlas_max")

    @unittest.skip("temporarily disabled")
    def test_Sizing(self):
        # arrange
        with AICSImage("img/img40_1.ome.tif") as image:
            max_edge = 2048
            # act
            atlas = generate_texture_atlas(
                im=image,
                prefix="test_Sizing",
                max_edge=max_edge,
                pack_order=[[3, 2, 1, 0], [4]],
            )
        atlas_maxedge = max(atlas.dims.atlas_width, atlas.dims.atlas_height)
        # assert
        self.assertTrue(atlas_maxedge <= max_edge)

    @unittest.skip("temporarily disabled")
    def test_pickChannels(self):
        packing_list = [[0], [1, 2], [3, 4]]
        # arrange
        with AICSImage("img/img40_1.ome.tif") as image:
            # act
            atlas = generate_texture_atlas(
                image, prefix="test_pickChannels", pack_order=packing_list
            )
            # returns as dict
            # metadata = atlas.get_metadata()
            # returns list of dicts
            image_dicts = atlas.atlas_list
            output_packed = [img.metadata["channels"] for img in image_dicts]
        # assert
        self.assertEqual(packing_list, output_packed)

    @unittest.skip("temporarily disabled")
    def test_metadata(self):
        packing_list = [[0], [1, 2], [3, 4]]
        prefix = "atlas"
        # arrange
        with AICSImage("img/img40_1.ome.tif") as image:
            # act
            atlas = generate_texture_atlas(
                image, prefix=prefix, pack_order=packing_list
            )
        # assert
        metadata = atlas.get_metadata()
        self.assertTrue(
            all(
                key in metadata
                for key in (
                    "tile_width",
                    "tile_height",
                    "width",
                    "height",
                    "channels",
                    "channel_names",
                    "tiles",
                    "rows",
                    "cols",
                    "atlas_width",
                    "atlas_height",
                    "images",
                )
            )
        )
        self.assertTrue(len(metadata["channel_names"]) == metadata["channels"])
