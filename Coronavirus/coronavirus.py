import os.path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
from sklearn.linear_model import LinearRegression
from pandas.plotting import register_matplotlib_converters
import seaborn as sns
from corona_libs import *

sns.set()
register_matplotlib_converters()

flags = True

country_ds = pd.DataFrame(
    columns=["name", "population", "filename", "flagname", "date_format",
             "color", "tot_count", "density", "tot_death", "lethality",
             "last_update"]
)
country_ds["name"] = [
    "United Kingdom", "Iran", "Italy", "Germany", "France", "China", "Spain",
    "Switzerland", "United States"
]
country_ds["population"] = [
    "66.44", "81.16", "60.48", "82.79", "66.99", "1386", "46.66", "8.57",
    "327.2"
]
country_ds["filename"] = [
    "UnitedKingdom.dat", "Iran.dat", "Italy.dat", "Germany.dat", "France.dat",
    "China.dat", "Spain.dat", "Switzerland.dat", "UnitedStates.dat"
]
country_ds["flagname"] = [
    "gb.png", "ir.png", "it.png", "de.png", "fr.png",
    "cn.png", "es.png", "ch.png", "us.png"
]
country_ds["date_format"] = [
    "%Y-%m-%d", None, "%d-%m-%Y", "%d.%m.%Y", "%d.%m.%Y", None, "%Y-%m-%d",
    "%Y-%m-%d", "%b %d %Y"
]
country_ds["color"] = [
    "purple", "brown", "green", "black", "blue", "red", "orange", "crimson",
    "darkblue"
]
country_ds.set_index("name", inplace=True)

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
for name in country_ds.index:
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

    ax1.plot(df.index, df["density"].values, ".",
             color=country_ds["color"].loc[name], label=label, markersize=8)
    ax1.plot(d1, 2**(model.predict(t1[:, np.newaxis])), "-.", alpha=0.6,
             color=country_ds["color"].loc[name], linewidth=0.8)
    ax1.plot(d2, 2**(model.predict(t2[:, np.newaxis])), "-.", alpha=0.3,
             color=country_ds["color"].loc[name], linewidth=0.8)

    df["death"].fillna(0, inplace=True)
    ax2.plot(df.index, df["lethality"].values, ".",
             color=country_ds["color"].loc[name], markersize=8)
    ax2.plot(df.index, df["lethality"].values, "-.",
             color=country_ds["color"].loc[name], alpha=0.3, linewidth=0.8)

ax1.text(0.01, 0.99, r"$\bf{Data Source:}$ " +
                     r"Wikipedia $\cdot$ 2019–20_coronavirus_pandemic",
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
ax1.set_ylim(bottom=country_ds["density"].min()/3)
ax1.set_ylim(top=country_ds["density"].max()*1.15)
ax1.yaxis.tick_right()
ax1.tick_params(axis="y", which="both", length=0)
ax1.set_ylabel("Infected per million people", rotation=270, labelpad=12)
ax1.yaxis.set_label_position("right")

ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(.5))
ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: "{:.0%}".format(y/100))
)
ax2.set_ylim((-0.25, country_ds["lethality"].max()*1.1))
ax2.set_ylabel("Lethality Rate", rotation=270, labelpad=7)
ax2.minorticks_on()
ax2.yaxis.tick_right()
ax2.tick_params(axis="y", which="both", length=0)
ax2.yaxis.set_label_position("right")

ax1.legend(loc="lower left", fontsize=8, ncol=3,
           bbox_to_anchor=(-0.004, 0.99, 1.008, 0.), mode="expand")
ax1.grid(b=True, which="major", linestyle="-")
ax1.grid(b=True, which="minor", linestyle="--")
ax2.grid(b=True, which="major", linestyle="-")
ax2.grid(b=True, which="minor", linestyle="--")

plt.tight_layout()

if flags is True:
    init_time = 0
    for idx, name in enumerate(country_ds.index):
        flagname = country_ds["flagname"].loc[name]
        if flagname is None or not os.path.isfile("flags/" + flagname):
            continue

        kwargs = {"delimiter": ";", "index_col": "data", "parse_dates": True}
        if country_ds["date_format"].loc[name] is not None:
            kwargs["date_parser"] = lambda x: pd.to_datetime(
                    x, format=country_ds["date_format"].loc[name])
        df = (pd.read_csv("datasets/" + country_ds["filename"]
              .loc[name], **kwargs))
        df["density"] = df["count"]/country_ds["population"].loc[name]

        t = mdates.date2num(df.index)

        if init_time == 0:
            init_time = t[-15]

        flagIdx = np.where(t == init_time)[0][0]

        plot_images2(t[flagIdx + idx], df["density"].values[flagIdx + idx],
                     "flags/" + flagname, ax=ax1)

plt.savefig("coronavirus.png", dpi=200)
plt.show()
