import numpy as np


def crop_img(img):

    inds = np.stack(np.where(img > 0))

    starts = np.min(inds, axis=1)
    ends = np.max(inds, axis=1) + 1

    croprange = [slice(s, e) for s, e in zip(starts, ends)]
    # dont crop the channel dimension
    croprange[0] = slice(0, None)

    img_out = img[croprange]

    return img_out, croprange
