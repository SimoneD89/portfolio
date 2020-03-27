import numpy as np
import matplotlib.image as image


def spacing(n):
    return "{:,d}".format(int(n)).replace(",", " ")


def plot_images(x, y, flagname, ax=None):
    ax = ax or plt.gca()

    image = plt.imread(flagname)
    img = []
    for i in range(image.shape[-1]):
        img.append(np.pad(image[:, :, i], ((7, 7), (7, 7))))
    image = np.swapaxes(np.swapaxes(np.array(img), 0, 1), 1, 2)

    for xi, yi in zip(x, y):
        im = OffsetImage(image, zoom=4/ax1.figure.dpi)
        im.image.axes = ax
        ab = AnnotationBbox(im, (xi, yi), frameon=False)
        ax.add_artist(ab)
    return None


def plot_images2(xi, yi, flagname, scale=.04, xshift=.0135, ax=None):
    ax = ax or plt.gca()

    # Flag coords are given in axes coords to avoid log scale distorsion
    # https://matplotlib.org/tutorials/advanced/transforms_tutorial.html
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
