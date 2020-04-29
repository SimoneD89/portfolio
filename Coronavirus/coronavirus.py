import os.path
import pandas as pd
from pandas.plotting import register_matplotlib_converters
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
from matplotlib.ticker import FormatStrFormatter
from matplotlib.offsetbox import OffsetImage, AnnotationBbox, Bbox
from matplotlib.transforms import TransformedBbox
from matplotlib.image import BboxImage
from matplotlib.legend_handler import HandlerBase
from sklearn.linear_model import LinearRegression
import seaborn as sns
sns.set()
register_matplotlib_converters()


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
        self.image_data = plt.imread(image_path)
        self.image_stretch = image_stretch


def plot_images(x, y, flagname, scale=4, xshift=0, alpha=None, ax=None):
    ax = ax or plt.gca()

    image = plt.imread(flagname)
    img = []
    for i in range(image.shape[-1]):
        img.append(np.pad(image[:, :, i], ((12, 12), (12, 12))))
    image = np.swapaxes(np.swapaxes(np.array(img), 0, 1), 1, 2)

    for xi, yi in zip(x, y):
        im = OffsetImage(image, zoom=scale/ax.figure.dpi, alpha=alpha)
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


def spacing(n):
    return "{:,d}".format(int(n)).replace(",", " ")


# 3 modes: 0 no flags, 1 representative flags, 2 flag markers
flags = 2

country_numbers = 9  # number of countries to display (max 9)

country_ds = pd.DataFrame(
    columns=["name", "population", "filename", "flagname", "date_format",
             "color", "tot_count", "density", "tot_death", "lethality",
             "last_update"]
)
country_ds["name"] = [
    "United Kingdom", "Lombardy", "Italy", "Germany", "France", "Ticino",
    "Spain", "Switzerland", "United States"
]
country_ds["population"] = [
    66.44, 10.06, 60.48, 82.79, 66.99, 0.3537, 46.66, 8.57, 327.2
]
country_ds["filename"] = [
    "UnitedKingdom.dat", "Lombardy.dat", "Italy.dat", "Germany.dat",
    "France.dat", "Ticino.dat", "Spain.dat", "Switzerland.dat",
    "UnitedStates.dat"
]
country_ds["flagname"] = [
    "gb.png", "it-lo.png", "it.png", "de.png", "fr.png",
    "ch-ti.png", "es.png", "ch.png", "us.png"
]
country_ds["date_format"] = [
    "%Y-%m-%d", "%d-%m-%Y", "%d-%m-%Y", "%d.%m.%Y", "%d.%m.%Y", "%Y-%m-%d",
    "%Y-%m-%d", "%Y-%m-%d", "%b %d %Y"
]
country_ds["color"] = [
    "purple", "teal", "green", "black", "blue", "red", "orange", "crimson",
    "darkblue"
]
country_ds.set_index("name", inplace=True)
country_ds = country_ds.iloc[:country_numbers]

for name in country_ds.index:
    kwargs = {"delimiter": ";", "index_col": "data", "parse_dates": True}
    if country_ds["date_format"].loc[name] is not None:
        kwargs["date_parser"] = lambda x: pd.to_datetime(
                x, format=country_ds["date_format"].loc[name])

    df = pd.read_csv("datasets/" + country_ds["filename"].loc[name], **kwargs)
    country_ds.loc[name, "tot_count"] = df["count"].iloc[-1]
    country_ds.loc[name, "tot_death"] = df["death"].iloc[-1]

country_ds["population"] = country_ds["population"].astype(float)
country_ds["tot_count"] = country_ds["tot_count"].astype(int)
country_ds["density"] = country_ds["tot_count"]/country_ds["population"]
country_ds["tot_death"] = country_ds["tot_death"].astype(int)
country_ds["lethality"] = country_ds["tot_death"]/country_ds["tot_count"]*100
country_ds["last_update"] = country_ds["tot_death"].astype("datetime64[ns]")
country_ds.sort_values(by=["density"], ascending=False, inplace=True)

plt.rcParams.update({"font.size": 12})

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                             figsize=(11.69, 8.27),
                             gridspec_kw={"height_ratios": [2, 1]})
f.subplots_adjust(hspace=0)

model = LinearRegression(fit_intercept=True)
init_time = 0
plots = []
handlers = {}
labels = []
for idx, name in enumerate(country_ds.index):
    kwargs = {"delimiter": ";", "index_col": "data", "parse_dates": True}
    if country_ds["date_format"].loc[name] is not None:
        kwargs["date_parser"] = lambda x: pd.to_datetime(
                x, format=country_ds["date_format"].loc[name])
    df = pd.read_csv("datasets/" + country_ds["filename"].loc[name], **kwargs)
    df["density"] = df["count"]/country_ds["population"].loc[name]
    df["lethality"] = df["death"]/df["count"]*100

    t = mdates.date2num(df.index)
    country_ds.loc[name, "last_update"] = df.index[-1]

    model.fit(t[-7:, np.newaxis], np.log2(df["density"].values[-7:]))

    t1 = np.linspace(t[-7:].min() - 0.5, t.max() + 0.25, 100)
    t2 = np.linspace(t[-1] - 15, t[-7:].min() - 0.5, 1000)
    d1 = mdates.num2date(t1)
    d2 = mdates.num2date(t2)

    label = (r"%s: $\tau \simeq$ %.1f d, " +
             r"tot = %s ($\dag\,$%s $\cdot$ %.1f%%)") % (
        name,
        1/model.coef_,
        spacing(country_ds["tot_count"].loc[name]),
        spacing(country_ds["tot_death"].loc[name]),
        country_ds["lethality"].loc[name]
    )
    labels.append(label)

    flagname = country_ds["flagname"].loc[name]
    if flagname is not None:
        image_path = "flags/" + flagname
    else:
        image_path = None

    plot = plt.plot([], [])
    if flags != 2 or flagname is None or not os.path.exists(image_path):
        plot = ax1.plot(df.index, df["density"].values, ".", markersize=8,
                        color=country_ds["color"].loc[name], label=label)
    ax1.plot(d1, 2**(model.predict(t1[:, np.newaxis])), "-.", alpha=0.6,
             color=country_ds["color"].loc[name], linewidth=0.8)
    ax1.plot(d2, 2**(model.predict(t2[:, np.newaxis])), "-.", alpha=0.3,
             color=country_ds["color"].loc[name], linewidth=0.8)
    plots.append(plot[0])

    if flags == 1 and flagname is not None and os.path.exists(image_path):
        if init_time == 0:
            init_time = t[-15]
        if os.path.isfile("flags/" + flagname):
            flagIdx = np.where(t == init_time)[0][0]
            plot_images([t[flagIdx + idx]],
                        [df["density"].values[flagIdx + idx]],
                        image_path, xshift=0.2, scale=6, ax=ax1)
    elif flags == 2 and flagname is not None and os.path.exists(image_path):
        custom_handler = ImageHandler()
        custom_handler.set_image(image_path, image_stretch=(-8, 2))
        handlers[plot[0]] = custom_handler
        plot_images(t[-15:], df["density"].values[-15:],
                    image_path, scale=2.7, ax=ax1)
        plot_images(t[-15:], df["lethality"].values[-15:],
                    image_path, scale=2.7, ax=ax2)

    df["death"].fillna(0, inplace=True)
    ax2.plot(df.index, df["lethality"].values, ".",
             color=country_ds["color"].loc[name], markersize=8)
    ax2.plot(df.index, df["lethality"].values, "-.",
             color=country_ds["color"].loc[name], alpha=0.3, linewidth=0.8)

ax1.text(0.01, 0.99, r"$\bf{Data Source:}$ " +
                     r"Wikipedia $\cdot$ 2019–20_coronavirus_pandemic" +
                     r", ti.ch",
         size=10, ha="left", va="top", clip_on=True, transform=ax1.transAxes)

ax1.text(0.982, 0.04, r"Model: $t \mapsto A\cdot 2^{(t-t_0)/\tau}$",
         size=10, ha="right", va="bottom", weight="bold", clip_on=True,
         bbox={"boxstyle": "round", "color": "blue", "alpha": 0.1},
         transform=ax1.transAxes)

last_update = mdates.date2num(country_ds["last_update"].max())
ax1.set_xlim(left=mdates.num2date(last_update - 14.5))
ax1.set_xlim(right=mdates.num2date(last_update + .5))

myFmt = mdates.DateFormatter("%b %d")

ax1.xaxis.set_major_formatter(myFmt)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

ax1.set_yscale("log")
ax1.set_ylim(bottom=country_ds["density"].min()/1.5)
ax1.set_ylim(top=country_ds["density"].max()*1.2)
ax1.yaxis.tick_right()
ax1.tick_params(axis="y", which="both", length=0)
ax1.tick_params(axis="y", which="major", pad=2)
ax1.tick_params(axis="y", which="minor", labelsize=6, pad=7)
ax1.yaxis.set_minor_formatter(
    ticker.FuncFormatter(lambda y, _: f"{str(y)[0]}")
)
ax1.set_ylabel("Infected per million people", rotation=270, labelpad=17)
ax1.yaxis.set_label_position("right")

ax2.yaxis.set_major_locator(ticker.MultipleLocator(5))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(1))
ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: "{:.0%}".format(y/100))
)
ax2.set_ylim((2, country_ds["lethality"].max()*1.1))
ax2.set_ylabel("Lethality rate", rotation=270, labelpad=12)
ax2.minorticks_on()
ax2.yaxis.tick_right()
ax2.tick_params(axis="y", which="both", length=0)
ax2.yaxis.set_label_position("right")

ax1.legend(plots, labels, handler_map=handlers, loc="lower left", ncol=3,
           fontsize=8, bbox_to_anchor=(-0.004, 0.99, 1.008, 0.), mode="expand")

ax1.grid(b=True, which="major", linestyle="-")
ax1.grid(b=True, which="minor", linestyle="--")
ax2.grid(b=True, which="major", linestyle="-")
ax2.grid(b=True, which="minor", linestyle="--")

plt.tight_layout()
plt.savefig("coronavirus.png", dpi=250)
plt.show()
