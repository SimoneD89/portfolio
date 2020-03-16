import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as ticker
import numpy as np
import datetime
import matplotlib.dates as mdates
from scipy.optimize import curve_fit
from scipy.special import erf
from pandas.plotting import register_matplotlib_converters
register_matplotlib_converters()
import seaborn as sns
sns.set()

def lin(t, b, t0, c):
    return (t-t0)/b + c


def spacing(n):
    return "{:,d}".format(int(n)).replace(",", " ")


country_ds = pd.DataFrame(
    columns=["name", "filename", "date_format", "color", "tot"]
)
country_ds["name"] = [
    "South Korea", "Iran", "Italy", "Germany", "France", "China", "Spain",
    "Switzerland", "United States"
]
country_ds["filename"] = [
    "SouthKorea.dat", "Iran.dat", "Italy.dat", "Germany.dat", "France.dat",
    "China.dat", "Spain.dat", "Switzerland.dat", "UnitedStates.dat"
]
country_ds["date_format"] = [
    "%Y-%m-%d", None, "%d-%m-%Y", "%d.%m.%Y", "%d.%m.%Y", None, "%Y-%m-%d",
    "%Y-%m-%d", "%b %d %Y"
]
country_ds["color"] = [
    "purple", "brown", "green", "orange", "blue", "red", "black", "crimson",
    "darkblue"
]

for i in country_ds.index:
    kwargs = {"delimiter": ";", "index_col": "data", "parse_dates": True}
    if country_ds["date_format"].iloc[i] is not None:
        kwargs["date_parser"] = lambda x: pd.to_datetime(
                x, format=country_ds["date_format"].iloc[i])

    df = pd.read_csv("datasets/" + country_ds["filename"].iloc[i], **kwargs)
    country_ds["tot"].iloc[i] = df["count"].iloc[-1]

country_ds["tot"] = country_ds["tot"].astype(int)
country_ds.sort_values(by=["tot"], ascending=False, inplace=True)
country_ds.reset_index(drop=True, inplace=True)

plt.rcParams.update({"font.size": 12})

f, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                             figsize=(11.69, 8.27),
                             gridspec_kw={"height_ratios": [2, 1]})
f.subplots_adjust(hspace=0)

for i in country_ds.index:
    kwargs = {"delimiter": ";", "index_col": "data", "parse_dates": True}
    if country_ds["date_format"].iloc[i] is not None:
        kwargs["date_parser"] = lambda x: pd.to_datetime(
                x, format=country_ds["date_format"].iloc[i])
    df = pd.read_csv("datasets/" + country_ds["filename"].iloc[i], **kwargs)

    t = mdates.date2num(df.index)
    param = [1, t[-7], 0]

    param, _ = curve_fit(lin, t[-7:],
                         np.log2(df["count"].values[-7:]),
                         p0=param)

    t1 = np.linspace(t[-7:].min() - 0.5, t.max() + 0.25, 100)
    t2 = np.linspace(t[-1] - 15, t[-7:].min() - 0.5, 1000)
    d1 = mdates.num2date(t1)
    d2 = mdates.num2date(t2)

    label = (r"%s: $\tau \simeq$ %.1f d, " +
             r"tot = %s ($\dag\,$%s $\cdot$ %.1f%%)") % (
        country_ds["name"].iloc[i],
        param[0],
        spacing(df["count"].values[-1]),
        spacing(df["death"].values[-1]),
        df["death"].values[-1]/df["count"].values[-1]*100
    )
    ax1.plot(df.index, df["count"].values, ".",
             color=country_ds["color"].iloc[i], label=label, markersize=8)
    ax1.plot(d1, 2**(lin(t1, *param)), "-.", alpha=0.6,
             color=country_ds["color"].iloc[i], linewidth=0.8)
    ax1.plot(d2, 2**(lin(t2, *param)), "-.", alpha=0.3,
             color=country_ds["color"].iloc[i], linewidth=0.8)

    df["death"].fillna(0, inplace=True)
    ax2.plot(df.index, (100*df["death"]/df["count"]).values, ".",
             color=country_ds["color"].iloc[i], markersize=8)
    ax2.plot(df.index, (100*df["death"]/df["count"]).values, "-.",
             color=country_ds["color"].iloc[i], alpha=0.3, linewidth=0.8)

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
ax1.set_ylim(bottom=country_ds["tot"].min()/6)
ax1.set_ylim(top=country_ds["tot"].max()*1.3)
ax1.yaxis.tick_right()
ax1.tick_params(axis="y", which="both", length=0)
ax1.set_ylabel("Infected People", rotation=270, labelpad=13)
ax1.yaxis.set_label_position("right")

ax2.yaxis.set_major_locator(ticker.MultipleLocator(2))
ax2.yaxis.set_minor_locator(ticker.MultipleLocator(.5))
ax2.yaxis.set_major_formatter(
    ticker.FuncFormatter(lambda y, _: "{:.0%}".format(y/100))
)
ax2.set_ylim((-0.25, 8))
ax2.set_ylabel("Lethality Rate", rotation=270, labelpad=16)
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
plt.savefig("coronavirus.png", dpi=200)
plt.show()
