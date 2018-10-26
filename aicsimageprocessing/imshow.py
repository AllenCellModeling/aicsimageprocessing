from .imgToProjection import imgtoprojection as im2proj
import matplotlib.pyplot as plt
import numpy as np


def imshow(img, scale=False, proj_method="max", colors=None, cmap="jet"):
    """
    Helper function to display a CZYX image in a jupyter notebook
    :param img: CZYX numpy array
    """

    from IPython.core.display import display
    import PIL.Image

    if colors is None:
        if img.shape[0] > 1:
            cmap = plt.get_cmap(cmap)
            colors = cmap(np.linspace(0, 1, img.shape[0]))
        else:
            colors = [[1, 1, 1, 1]]

    img = im2proj(
        img,
        proj_all=True,
        proj_method=proj_method,
        local_adjust=True,
        global_adjust=True,
        colors=colors,
    )

    if scale:
        scale_amt = 255 / np.max(img)

    display(
        PIL.Image.fromarray(
            ((img * (scale_amt if scale else 1)).astype("uint8")).transpose(1, 2, 0),
            "RGB",
        )
    )
