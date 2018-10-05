import numpy as np
import aicsimageprocessing as proc
import skfmm
from scipy.ndimage.morphology import distance_transform_edt as bwdist
import math
from skimage.measure import regionprops

def imgtocoords(im_cell, im_nuc, major_angle='cell'):
    """
    Return unit vector of v
    :param im_cell: zyx binary image of a cell shape
    :param im_nuc: zyx binary image of a nuclear shape

    :return: im_ratio - channels corresponding to ratio image (1 at cell boundary, 0 on nuclear boundary, -1 inside nucleus)
             im_th - spherical coordinate system theta (radians)
             im_phi - spherical coordinate system phi (radians)
             im_r - radial distance from center of nucleus
             major_angle - angle of cell or nuclear shape (radians)
    """
    
    cell_dist_in = skfmm.distance(np.ma.MaskedArray(im_cell, im_nuc)).data
    cell_dist_out = skfmm.distance(np.ma.MaskedArray(im_nuc==0, im_cell==0)).data
    
    nuc_dist = bwdist(im_nuc)
    nuc_ratio = -nuc_dist/np.max(nuc_dist)
    nuc_ratio[np.isnan(nuc_ratio)] = 0


    cyto_ratio = cell_dist_out/(cell_dist_out+cell_dist_in)
    cyto_ratio[np.isnan(cyto_ratio)] = 0

    im_cyto = (im_cell>0) & (im_nuc==0)

    im_ratio = np.zeros(im_nuc.shape)
    im_ratio[im_nuc>0] = nuc_ratio[im_nuc>0]
    im_ratio[im_cyto] = cyto_ratio[im_cyto]


    centroid = regionprops((im_nuc>0).astype('uint8'))[0]['centroid']

    if major_angle == 'nuc':
        mangle = proc.get_major_angle(im_nuc, degrees_or_radians = 'radians')[0][1]
    elif major_angle == 'cell':
        mangle = proc.get_major_angle(im_cell, degrees_or_radians = 'radians')[0][1]

    coords = np.where(im_cell>0)


    th, phi, r = proc.cart2sph(coords[1] - centroid[1], coords[2] - centroid[2], coords[0] - centroid[0])

    th = th - mangle
    th = th - 2*math.pi*np.floor((th+math.pi)/(2*math.pi))

    im_th = np.zeros(im_nuc.shape)
    im_th[im_cell>0] = th

    im_phi = np.zeros(im_nuc.shape)
    im_phi[im_cell>0] = phi

    im_r = np.zeros(im_nuc.shape)
    im_r[im_cell>0] = r

    return im_ratio, im_th, im_phi, im_r, major_angle