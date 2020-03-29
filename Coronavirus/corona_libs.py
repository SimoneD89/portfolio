import os.path
import numpy as np
import matplotlib.image as image
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, Bbox
from matplotlib.transforms import TransformedBbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from matplotlib._png import read_png


def spacing(n):
    return "{:,d}".format(int(n)).replace(",", " ")


def plot_images(x, y, flagname, scale=4, xshift=0, ax=None):
    ax = ax or plt.gca()

    image = plt.imread(flagname)
    img = []
    for i in range(image.shape[-1]):
        img.append(np.pad(image[:, :, i], ((10, 10), (10, 10))))
    image = np.swapaxes(np.swapaxes(np.array(img), 0, 1), 1, 2)

    for xi, yi in zip(x, y):
        im = OffsetImage(image, zoom=scale/ax.figure.dpi)
        im.image.axes = ax
        ab = AnnotationBbox(im, (xi + xshift, yi), frameon=False)
        ax.add_artist(ab)
    return None


def plot_images2(xi, yi, flagname, scale=.04, xshift=.0135, ax=None):
    ax = ax or plt.gca()

    # Flag coords are given in axes coords to avoid log scale distorsion
    # https://matplotlib.org/tutorials/advanced/transforms_tutorial.html
    # It needs to be executed after tight_layout()
    flagCoord = ax.transData.transform((xi, yi))
    flagCoord = ax.transAxes.inverted().transform(flagCoord)

    figW, figH = ax.get_figure().get_size_inches()
    _, _, w, h = ax.get_position().bounds
    disp_ratio = figH*h/(figW*w)

    im = image.imread(flagname)
    ax.imshow(im, aspect="auto", zorder=10, transform=ax.transAxes,
              extent=(flagCoord[0] + xshift - scale/2*disp_ratio,
                      flagCoord[0] + xshift + scale/2*disp_ratio,
                      flagCoord[1] - scale/2,
                      flagCoord[1] + scale/2))
    return None


# stackoverflow.com/questions/26029592/insert-image-in-matplotlib-legend
class ImageHandler(HandlerBase):
    def create_artists(self, legend, orig_handle,
                       xdescent, ydescent, width, height, fontsize,
                       trans):

        # enlarge the image by these margins
        sx, sy = self.image_stretch

        # create a bounding box to house the image
        bb = Bbox.from_bounds(xdescent - sx,
                              ydescent - sy,
                              width + sx,
                              height + sy)

        tbb = TransformedBbox(bb, trans)
        image = BboxImage(tbb)
        image.set_data(self.image_data)

        self.update_prop(image, orig_handle, legend)

        return [image]

    def set_image(self, image_path, image_stretch=(0, 0)):
        if not os.path.exists(image_path):
            sample = get_sample_data("grace_hopper.png", asfileobj=False)
            self.image_data = read_png(sample)
        else:
            self.image_data = read_png(image_path)

        self.image_stretch = image_stretch
