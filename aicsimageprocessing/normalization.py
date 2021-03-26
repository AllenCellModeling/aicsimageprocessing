import numpy as np
import scipy.ndimage


def normalize_img(img, mask=None, method="img_bg_sub", n_dims=None, lower=None,
                  upper=None):
    if n_dims is None:
        if img.shape[0] < 10:
            # assume first dimension is channels
            n_dims = len(img.shape) - 1

    # if single channel
    if n_dims == len(img.shape):
        if (upper is not None) or (lower is not None):
            if not isinstance(upper, (int, float)) or not isinstance(upper, (int, float)):
                raise ValueError("When normalizing a single channel, `upper` and
                                  `lower` must each be None or a single numeric
                                  value.")

        return normalize_channel(img, mask, method, upper, lower)

    if (upper is None) or (lower is None):
        upper = [None] * img.shape[0]
        lower = [None] * img.shape[0]

    if isinstance(upper, (int, float)):
        upper = [upper] * img.shape[0]
    if isinstance(lower, (int, float)):
        lower = [lower] * img.shape[0]

    for channel_ix, (channel, upper, lower) in enumerate(zip(img, upper, lower)):
        img[channel_ix] = normalize_channel(channel, mask, method, upper, lower)

    return img


def normalize_channel(img, mask=None, method="img_bg_sub", lower=None,
                      upper=None):

    img = img.astype(float)

    if mask is None:
        mask = np.ones(img.shape)

    if method == "img_bg_sub":
        im_f = scipy.ndimage.gaussian_filter(img, sigma=0.5)
        prct = np.percentile(im_f[mask > 0], 50)

        im_f = im_f - prct
        im_out = img - prct

        im_f[im_f < 0] = 0
        im_out[im_out < 0] = 0

        im_out = im_out / np.max(im_f)

        im_out[im_out < 0] = 0
        im_out[im_out > 1] = 1

    elif method == "trans":  # transmitted
        # Normalizes to 0.5, with std of 0.1
        im_f = scipy.ndimage.gaussian_filter(img, sigma=0.5)
        mu = np.mean(im_f)
        std = np.std(im_f)

        std_fract = 10

        im_out = (img - mu) / (std * std_fract) + 0.5

        im_out[im_out < 0] = 0
        im_out[im_out > 1] = 1
    elif method == "cvapipe":
        lower = np.percentile(img, 0.05)
        upper = np.percentile(img, 99.5)
        im_out = rescale(img, lower, upper)
    elif method == "custom_scale":
        im_out = rescale(img, lower, upper)
    elif method == "custom_percentile":
        lower = np.percentile(img, lower)
        upper = np.percentile(img, upper)
        im_out = rescale(img, lower, upper)
    elif method is None or method == "none":
        pass
    else:
        raise NotImplementedError

    return im_out


def rescale(img, lower, upper):
    img[img > upper] = upper
    img[img < lower] = lower
    return (img - lower) / (upper - lower)


def mask_normalization(image, mask, method):
    raise NotImplementedError
