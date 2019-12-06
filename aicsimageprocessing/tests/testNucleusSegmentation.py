#!/usr/bin/env python

# author: Zach Crabtree zacharyc@alleninstitute.org

import unittest

from aicsimageio.readers import TiffReader
from aicsimageprocessing.segmentation import nucleusSegmentation


class TestNucleusSegmentation(unittest.TestCase):

    @staticmethod
    @unittest.skip("temporarily disabled")
    def test_Segmentation():
        cell_index_im = TiffReader("img/segmentation/input_1_cellWholeIndex.tiff").load()
        original_im = TiffReader("img/segmentation/input_3_nuc_orig_img.tiff").load()

        nucleusSegmentation.fill_nucleus_segmentation(cell_index_im, original_im)
