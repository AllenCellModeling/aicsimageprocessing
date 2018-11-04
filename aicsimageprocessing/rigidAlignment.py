import numpy as np
import sklearn.decomposition
import scipy
from .alignMajor import align_major, angle_between

# img - a CYXZ numpy array, channel order is generally [DNA, NUCLEUS, ... ]


def cell_rigid_registration(img, ch_crop=1, ch_angle=1, ch_com=0, ch_flipdim=1, bbox_size=None):
    # If bbox_size is not None ch_crop is ignored
    # rotate
    angle = get_major_angle(get_channel(img, ch_angle))

    angle = [[i, -j] for i, j in angle]

    _, croprange = crop_img(get_channel(img, ch_crop), method="bigger")

    img = img[croprange]

    img = align_major(img, angle)

    com = np.floor(get_center_of_mass(get_channel(img, ch_com))+0.5)

    # make sure we symmetrically crop around the COM
    if bbox_size is None:
        _, croprange = crop_img(get_channel(img, ch_crop))

        ranges = np.array([[s.start, s.stop] for s in croprange])
        padranges = (com[1:] - ranges[1:, 1]) - (ranges[1:, 0] - com[1:])

        pad_pre = np.abs(padranges * (padranges < 0)).astype(int)
        pad_post = np.abs(padranges * (padranges > 0)).astype(int)

        ranges[1:, 0] -= pad_pre
        ranges[1:, 1] += pad_post

    else:
        bbox_size = np.array(bbox_size)
        pad_size = bbox_size/2
        ranges = np.transpose(np.vstack([com - pad_size, com + pad_size]))
        ranges = np.floor(ranges+0.5).astype(int).astype(object)
        ranges[0] = [0, None]

    pad_pre = np.hstack([0, ranges[1:, 0]])
    pad_pre = np.abs(pad_pre * (pad_pre < 0))

    pad_post = np.hstack([0, ranges[1:, 1] - np.array(img.shape)[1:]])
    pad_post = np.abs(pad_post * (pad_post > 0))

    pad_width = np.transpose(np.vstack([pad_pre, pad_post])).astype(int)

    non_nones = ~np.equal(ranges, None)
    range_vals = ranges[non_nones]
    ranges[non_nones] = range_vals * (range_vals > 0)

    croprange = [slice(s, e) for s, e in ranges]

    img = img[croprange]

    img = np.pad(img, pad_width, mode='constant', constant_values=0)

    # flipdim
    flipdim = get_flipdims(get_channel(img, ch_flipdim))
    img = flipdims(img, flipdim)

    return img, angle, flipdim


def cell_rigid_deregistration(img, flipdim_orig, angle_orig, com_orig, imsize_orig, ch_crop=1, ch_com=0):
    # deflip the image
    img = flipdims(img, flipdim_orig)

    # derotate the image
    angle = [[i, -j] for i, j in angle_orig]
    img = align_major(img, angle)

    # depad the image
    img = pad_to_position(img, ch_crop, ch_com, com_orig, imsize_orig)

    return img


def get_channel(img, channel):
    return np.expand_dims(img[channel], 0)


def get_rigid_reg_stats(img, com_method='nuc'):
    imsize = img.shape
    com = get_center_of_mass(img, com_method)

    return imsize, com, angle, flipdim


def get_major_angle(img, degrees_or_radians="degrees"):
    # align on the 2D projection
    if len(img.shape) == 4:
        img = np.sum(img, axis=3)

    pos = np.stack(np.where(img > 0), axis=1)
    pca = sklearn.decomposition.PCA()
    pca.fit(pos - np.mean(pos, axis=0))
    angles = np.array(pca.components_[0, 1:3])
    angle = angle_between(angles, np.array([1, 0]))
    if angles[1] < 0:
        angle = 360 - angle
    if degrees_or_radians == "radians":
        angle = angle * 0.0174533
    return [[int(0), angle]]


def flipdims(img, flipdim):
    for flip, i in zip(flipdim, range(len(flipdim))):
        if flip:
            img = np.flip(img, i)

    # dont flip on z for the time being
    flipdim[-1] = 0

    return img


def get_center_of_mass(img):

    com = np.mean(np.stack(np.where(img > 0)), axis=1)

    return com


def crop_img(img, method='tight'):

    inds = np.stack(np.where(img > 0))

    starts = np.min(inds, axis=1)
    ends = np.max(inds, axis=1)+1

    if method == 'bigger':
        width = ends - starts

        starts_tmp = starts - width
        starts_tmp[starts_tmp < 0] = 0

        ends_tmp = ends + width

        starts[1:3] = starts_tmp[1:3]
        ends[1:3] = ends_tmp[1:3]

        starts[-1] = 0
        ends[-1] = img.shape[-1]

    croprange = [slice(s, e) for s, e in zip(starts, ends)]
    # dont crop the channel dimension
    croprange[0] = slice(0, None)

    img_out = img[croprange]

    return img_out, croprange


def get_flipdims(img):
    skew = scipy.stats.skew(np.stack(np.where(img), axis=0), axis=1)
    skew[-1] = 0

    return skew < 0


def pad_to_position(img, ch_crop, ch_com, com_target, imsize_target):

    _, croprange_pt2 = crop_img(get_channel(img, ch_crop))
    img = img[croprange_pt2]

    com = get_center_of_mass(get_channel(img, ch_com))

    pad_com = com-com_target

    pad_pre = (com_target - (com + 1))[1:]
    pad_post = (imsize_target - com_target - (np.array(img.shape) - (com + 1)))[1:]

    pad_width = [[0, 0]]

    for pre, post in zip(pad_pre, pad_post):
        pad_width += [[int(np.ceil(pre)), int(np.floor(post))]]

    img_out = np.pad(img, pad_width, mode='constant', constant_values=0)

    return img_out


def pad_to_center(img, com):
    _, croprange_pt2 = crop_img(get_channel(img, ch_crop))
    img = img[croprange_pt2]

    com = get_center_of_mass(get_channel(img, ch_com))

    pad_dims = img.shape - (com+1) - com

    img = pad_to_com(img, pad_dims)

    return img


def pad_to_com(img, pad_dims):

    pad_dims = pad_dims.astype(int)

    # skip the channel dimension
    pad_width = [[0, 0]]

    for i in pad_dims[1:]:
        if i > 0:
            pad = [[np.abs(i), 0]]
        else:
            pad = [[0, np.abs(i)]]

        pad_width += pad

    img = np.pad(img, pad_width, mode='constant', constant_values=0)

    return img
