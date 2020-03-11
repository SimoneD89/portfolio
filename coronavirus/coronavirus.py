import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import datetime
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.special import erf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()


def lin(t, b, t0, c):
    return (t-t0)/b + c


country_ds = pd.DataFrame(
    columns=["name", "filename", "date_format", "color", "fit", "tot"]
)
country_ds["name"] = [
    "South Korea", "Iran", "Italy", "Germany", "France", "China", "Spain"
]
country_ds["filename"] = [
    "SouthKorea.dat", "Iran.dat", "Italy.dat", "Germany.dat", "France.dat",
    "China.dat", "Spain.dat"
]
country_ds["date_format"] = [
    "%Y-%m-%d", None, "%d-%m-%Y", "%d.%m.%Y", "%d.%m.%Y", None, "%Y-%m-%d"
]
country_ds["color"] = [
    "purple", "brown", "green", "orange", "blue", "red", "black"
]
country_ds["fit"] = [
    "lin", "lin", "lin", "lin", "lin", "lin", "lin"
]

for i in country_ds.index:
    if country_ds["date_format"].iloc[i] is None:
        df = pd.read_csv(
            "datasets/" + country_ds["filename"].iloc[i],
            delimiter=";",
            index_col="data",
            parse_dates=True,
        )
    else:
        df = pd.read_csv(
            "datasets/" + country_ds["filename"].iloc[i],
            delimiter=";",
            index_col="data",
            parse_dates=True,
            date_parser=lambda x: pd.to_datetime(
                x, format=country_ds["date_format"].iloc[i]
            )
        )
    country_ds["tot"].iloc[i] = df["count"].iloc[-1]

country_ds.sort_values(by=["tot"], ascending=False, inplace=True)
country_ds.reset_index(drop=True, inplace=True)

plt.rcParams.update({"font.size": 12})
plt.style.use("ggplot")

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                             figsize=(11.69, 8.27),
                             gridspec_kw={"height_ratios": [2, 1]})
f.subplots_adjust(hspace=0)

for i in country_ds.index:
    if country_ds["date_format"].iloc[i] is None:
        df = pd.read_csv(
            "datasets/" + country_ds["filename"].iloc[i],
            delimiter=";",
            index_col="data",
            parse_dates=True,
        )
    else:
        df = pd.read_csv(
            "datasets/" + country_ds["filename"].iloc[i],
            delimiter=";",
            index_col="data",
            parse_dates=True,
            date_parser=lambda x: pd.to_datetime(
                x,
                format=country_ds["date_format"].iloc[i])
        )

    t = mdates.date2num(df.index)
    func = lin
    param = [1, t[-7], 0]

    param, _ = curve_fit(func, t[-7:],
                         np.log2(df["count"].values[-7:]),
                         p0=param)

    t1 = np.linspace(t[-7:].min() - 0.5, t.max() + 0.25, 100)
    t2 = np.linspace(t[-1] - 15, t[-7:].min() - 0.5, 1000)
    d1 = mdates.num2date(t1)
    d2 = mdates.num2date(t2)

    label = r"{0!s}: $\tau \simeq$ {1:.1f} d, ".format(
        country_ds["name"].iloc[i], param[0]
    ) + "tot = {0:,d}".format(
        df["count"].values[-1]
    ).replace(",", " ") + r" ($\dag\,${:,d}".format(
        int(df["death"].values[-1])
    ).replace(",", " ") + r" $\cdot$ {:.1f}%)".format(
        df["death"].values[-1]/df["count"].values[-1]*100
    )
    ax1.plot(df.index, df["count"].values, ".",
             color=country_ds["color"].iloc[i], label=label, markersize=10)
    ax1.plot(d1, 2**(func(t1, *param)), ":", alpha=0.7,
             color=country_ds["color"].iloc[i])
    ax1.plot(d2, 2**(func(t2, *param)), ":", alpha=0.4,
             color=country_ds["color"].iloc[i])

    ax2.plot(df.index, df["death"].values, ":.",
             color=country_ds["color"].iloc[i], markersize=10)

ax1.text(0.982, 0.04, r"Model: $t \mapsto A\cdot 2^{(t-t_0)/\tau}$",
         size=10, ha="right", va="bottom", weight="bold", clip_on=True,
         bbox={"boxstyle": "round", "color": "blue", "alpha": 0.1},
         transform=ax1.transAxes)

ax1.set_xlim(left=mdates.num2date(t[-1] - 14.5))
ax1.set_xlim(right=plt.xlim()[1] - 2.25)

myFmt = mdates.DateFormatter("%b %d")

ax1.xaxis.set_major_formatter(myFmt)
ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
ax1.xaxis.set_minor_locator(mdates.DayLocator())

ax1.set_yscale("log")
ax1.set_ylim(bottom=50)
ax1.yaxis.tick_right()
ax1.set_ylabel("Infected People", rotation=270, labelpad=13)
ax1.yaxis.set_label_position("right")

ax2.set_yscale("log")
ax2.yaxis.tick_right()
ax2.set_ylabel("Dead People", rotation=270, labelpad=13)
ax2.yaxis.set_label_position("right")

ax1.legend(loc="upper left", bbox_to_anchor=(0, 0.9))
ax1.grid(b=True, which="major", linestyle="-")
ax1.grid(b=True, which="minor", linestyle="--")
ax2.grid(b=True, which="major", linestyle="-")
ax2.grid(b=True, which="minor", linestyle="--")

plt.tight_layout()
plt.savefig("coronavirus.png", dpi=150)
plt.show()
