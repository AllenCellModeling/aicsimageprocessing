# Author: Evan Wiederspan <evanw@alleninstitute.org>

import unittest
from random import sample
import numpy as np
from aicsimageprocessing.backgroundCrop import crop, get_edges


class TestBackgroundCrop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.testImage = np.zeros((3, 10, 10, 10))
        cls.testImage[:, 3:6, 3:6, 3:6] = np.eye(3)

    def test_cropInputs(self):
        with self.assertRaises(ValueError, msg="Positive axis must be in range"):
            crop(self.testImage, axis=(1, 2, 10))
        with self.assertRaises(ValueError, msg="Negative axis must be in range"):
            crop(self.testImage, axis=(1, 2, -10))
        with self.assertRaises(ValueError, msg="Axis must be iterable"):
            crop(self.testImage, axis=0)

    def test_cropOutputs(self):
        self.assertEqual(
            crop(self.testImage, 2).shape,
            (3, 10, 10, 10),
            "Nothing cropped when crop value not found",
        )
        self.assertEqual(
            crop(self.testImage).shape,
            (3, 3, 3, 3),
            "Cropped output is in expected shape",
        )

    def test_getEdges(self):
        # run test 5 times
        for _ in range(5):
            test = np.zeros((3, 11, 11, 11))
            # put in random blob of 1's
            ends = tuple(sorted(sample(range(0, 10), 2)) for _ in range(3))
            if not all(right - left > 1 for left, right in ends):
                continue
            test[tuple([slice(None, None)] + list(slice(*e) for e in ends))] = 1
            edges = get_edges(test)
            self.assertEqual(
                ends, edges, "Edges " + str(edges) + " found for " + str(ends)
            )

    def test_getCropSlices(self):
        cropped, slice_indices = crop(self.testImage, 0, get_slices=True)
        self.assertEqual(
            cropped.shape,
            (3, 3, 3, 3),
            "Cropped output is expected shape when returning slices",
        )
        self.assertEqual(
            slice_indices,
            ([0, 3], [3, 6], [3, 6], [3, 6]),
            "Slice indices match expected values",
        )

    def test_cropAxis(self):
        self.assertEqual(
            crop(self.testImage, 0, (-2, -1)).shape,
            (3, 10, 3, 3),
            "Crop only specified axes",
        )
        self.assertEqual(
            crop(self.testImage, 0, tuple()).shape,
            (3, 10, 10, 10),
            "Crop skipped when no axis specified",
        )

    def test_cropPadding(self):
        self.assertEqual(
            crop(self.testImage, 0, padding=1).shape,
            (3, 5, 5, 5),
            "Crops to expected shape with padding",
        )
        self.assertEqual(
            crop(self.testImage, 0, padding=100).shape,
            (3, 10, 10, 10),
            "Crops correctly with large padding",
        )
