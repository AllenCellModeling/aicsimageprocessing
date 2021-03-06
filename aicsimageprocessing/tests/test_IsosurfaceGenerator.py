import math as m
import unittest

import numpy as np
from aicsimageio import AICSImage

from aicsimageprocessing import isosurfaceGenerator


class TestIsosurfaceGenerator(unittest.TestCase):
    def setUp(self):
        unittest.TestCase.__init__(self)

    def runTest(self):
        # this method has to be included in a testgroup in order for it be run
        self.assertTrue(True)

    @staticmethod
    @unittest.skip("temporarily disabled")
    def testSphere(radius=30):
        bounding_cube = np.zeros(
            (radius * 2 + 3, radius * 2 + 3, radius * 2 + 3)
        ).astype(np.float32)
        center = bounding_cube.shape[0] // 2
        for x in range(bounding_cube.shape[0]):
            for y in range(bounding_cube.shape[1]):
                for z in range(bounding_cube.shape[2]):
                    x_dist = (x - center) ** 2
                    y_dist = (y - center) ** 2
                    z_dist = (z - center) ** 2
                    distance_from_center = m.sqrt(x_dist + y_dist + z_dist)
                    if distance_from_center <= radius:
                        bounding_cube[x, y, z] = 1
        with AICSImage(bounding_cube, dims="XYZ") as sphere:
            mesh = isosurfaceGenerator.generate_mesh(sphere, isovalue=0.99)

        mesh.save_as_obj("img/test_sphere.obj")

    @staticmethod
    @unittest.skip("temporarily disabled")
    def testCube(size=3):
        # these cubes appear to have strangely beveled edges but I think that is an
        # artifact of the
        # linear interpolation between values of the marching cubes algorithm.
        cube = np.zeros((size, size, size))
        # set all pixels/voxels between 1 and size-1 to 1
        # (leave a single row/col buffer around entire cube shape)
        cube[1:-1, 1:-1, 1:-1] = 1
        with AICSImage(cube, dims="XYZ") as cubeimage:
            mesh = isosurfaceGenerator.generate_mesh(cubeimage, isovalue=0)
        mesh.save_as_obj("img/test_cube.obj")

    @staticmethod
    @unittest.skip("temporarily disabled")
    def testCellImage():
        with AICSImage("./img/img40_1.ome.tif") as cell_image:
            mesh = isosurfaceGenerator.generate_mesh(cell_image, channel=4)
        mesh.save_as_obj("img/test_file.obj")
