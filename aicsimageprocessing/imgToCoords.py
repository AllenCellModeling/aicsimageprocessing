import numpy as np
from . import rigidAlignment
import skfmm
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import math
from skimage.measure import regionprops


def img_to_coords(im_cell, im_nuc, major_angle_object="cell"):
    """
    Return unit vector of v
    :param im_cell: zyx binary image of a cell shape
    :param im_nuc: zyx binary image of a nuclear shape
    :major_angle_object: string that specifies from which object the major angle is determined can be 'cell' or 'nuc'

    :return: im_ratio - channels corresponding to ratio image (1 at cell boundary, 0 on nuclear boundary, -1 inside nucleus)
             im_th - spherical coordinate system theta (radians)
             im_phi - spherical coordinate system phi (radians)
             im_r - radial distance from center of nucleus
             major_angle - angle of cell or nuclear shape (radians)
    """

    cell_dist_in = skfmm.distance(np.ma.MaskedArray(im_cell, im_nuc)).data
    cell_dist_out = skfmm.distance(np.ma.MaskedArray(im_nuc == 0, im_cell == 0)).data

    nuc_dist = bwdist(im_nuc)
    nuc_ratio = -nuc_dist / np.max(nuc_dist)
    nuc_ratio[np.isnan(nuc_ratio)] = 0

    cyto_ratio = cell_dist_out / (cell_dist_out + cell_dist_in)
    cyto_ratio[np.isnan(cyto_ratio)] = 0

    im_cyto = (im_cell > 0) & (im_nuc == 0)

    im_ratio = np.zeros(im_nuc.shape)
    im_ratio[im_nuc > 0] = nuc_ratio[im_nuc > 0]
    im_ratio[im_cyto] = cyto_ratio[im_cyto]

    centroid = regionprops((im_nuc > 0).astype("uint8"))[0]["centroid"]

    if major_angle_object == "nuc":
        major_angle = rigidAlignment.get_major_angle(im_nuc, degrees_or_radians="radians")[0][1]
    elif major_angle_object == "cell":
        major_angle = rigidAlignment.get_major_angle(im_cell, degrees_or_radians="radians")[0][1]

    coords = np.where(im_cell > 0)

    th, phi, r = cart2sph(
        coords[1] - centroid[1], coords[2] - centroid[2], coords[0] - centroid[0]
    )

    th = th - major_angle
    th = th - 2 * math.pi * np.floor((th + math.pi) / (2 * math.pi))

    im_th = np.zeros(im_nuc.shape)
    im_th[im_cell > 0] = th

    im_phi = np.zeros(im_nuc.shape)
    im_phi[im_cell > 0] = phi

    im_r = np.zeros(im_nuc.shape)
    im_r[im_cell > 0] = r

    return im_ratio, im_th, im_phi, im_r, major_angle


rads_per_deg = 0.0174533


def cart2sph(x, y, z, degrees_or_radians="radians"):
    # Angles are in radians

    azimuth = np.arctan2(y, x)
    elevation = np.arctan2(z, np.sqrt(x ** 2 + y ** 2))
    r = np.sqrt(x ** 2 + y ** 2 + z ** 2)

    if degrees_or_radians == "degrees":
        azimuth = azimuth / rads_per_deg
        elevation = elevation / rads_per_deg

    return azimuth, elevation, r


def sph2cart(azimuth, elevation, r, degrees_or_radians="radians"):
    # Angles are in radians

    if degrees_or_radians == "degrees":
        azimuth = azimuth * rads_per_deg
        elevation = elevation * rads_per_deg

    x = r * np.cos(elevation) * np.cos(azimuth)
    y = r * np.cos(elevation) * np.sin(azimuth)
    z = r * np.sin(elevation)
    return x, y, z
