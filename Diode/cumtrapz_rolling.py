import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd

from datetime import datetime
import pickle
import configparser
import re

from os import listdir
from os.path import isfile, join, dirname
import sys

from scipy import integrate
import scipy.optimize as optimization
from matplotlib.lines import Line2D


config = configparser.ConfigParser()
config.read(join(dirname(sys.argv[0]), "config.ini"))
# config["Paths"]["path_reports"] = "reports"
# config["Paths"]["path_datasets"] = "datasets"
# with open('config.ini', 'w') as configfile:
#     config.write(configfile)
#
# --config.ini--
# [Paths]
# path_reports: reports
# path_datasets: datasets


def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def str2list(string):
    rex = re.compile(r"^\s*-?\s*[0-9]+\s*((-|,)\s*-?\s*[0-9]+\s*)*$")
    if rex.match(string.strip()):
        string = list(string.replace(" ", "").split(","))
        m = []
        for i in string:
            ind = [j for j, x in enumerate(i.split("-")) if x == ""]
            if ind == [] and len(i.split("-")) == 2:
                m.extend(range(int(i.split("-")[0]), int(i.split("-")[1])+1))
            elif ind == [0, 2]:
                m.extend(range(-int(i.split("-")[1]), -int(i.split("-")[3])+1))
            elif ind == [0] and len(i.split("-")) == 3:
                m.extend(range(-int(i.split("-")[1]), int(i.split("-")[2])+1))
            elif ind == [1]:
                m.extend(range(int(i.split("-")[0]), -int(i.split("-")[2])+1))
            else:
                m.append(int(i))
        return sorted(set(m))
    else:
        return []


def list2str(lst):
    if not all([isinstance(lst, list), lst,
                *[isinstance(x, int) for x in lst]]):
        return ""
    elif len(lst) == 1:
        return str(lst[0])

    lst, string, m, ran = sorted(list(dict.fromkeys(lst))), "", lst[0], False
    for i in range(len(lst)-1):
        if lst[i+1] - lst[i] != 1:
            if ran:
                string += str(m) + "-" + str(lst[i]) + ", "
                ran = False
            else:
                string += str(lst[i]) + ", "
            m = lst[i+1]
        else:
            ran = True
    string += str(m) + "-" + str(lst[i+1]) if ran else str(lst[i+1])
    return string


def intersection(p1, p2, x=None, y=None):
    if x is None and y is not None:
        return (p1[0] + (y-p1[1])/(p2[1]-p1[1])*(p2[0]-p1[0]), y)
    elif y is None and x is not None:
        return intersection(p1[::-1], p2[::-1], y=x)[::-1]
    else:
        return ()


def circle_plot(ids, kind, x, y, size=(6, 5), ax=None, legend=True, **kwargs):
    use_ax = True
    if ax is None:
        fig = plt.figure(figsize=size)
        ax = plt.gca()
        use_ax = False

    colors = dict(zip(sorted(set(kind)),
                      ["royalblue", "green", "red",
                       "orange", "purple", "grey"]))

    for i in range(len(ids)):
        pad = 0.5 if ids[i] < 10 else 0.3 if ids[i] < 100 else 0
        ax.text(x[i], y[i], str(ids[i]), size=8, ha="center", va="center",
                color="white", weight="bold", clip_on=True,
                bbox={"boxstyle": "circle", "color": colors[kind[i]],
                      "alpha": 0.8, "pad": pad})

    if len(set(x)) == 1:
        ax.set_xlim(0, 2*max(x))
    else:
        ax.set_xlim(min(x) - .2/size[0]*(max(x)-min(x)),
                    max(x) + .2/size[0]*(max(x)-min(x)))

    if len(set(y)) == 1:
        ax.set_ylim(0, 2*max(y))
    else:
        ax.set_ylim(min(y) - .2/size[1]*(max(y)-min(y)),
                    max(y) + .2/size[1]*(max(y)-min(y)))

    ax.grid(b=True, which="major", linestyle="-")
    ax.grid(b=True, which="minor", linestyle="--")
    ax.minorticks_on()
    if legend:
        labels = ["%s" % str(i) for i in colors.keys()]
        lines = [Line2D([0], [0], marker="o", color="w", label="Circle",
                 markerfacecolor=c, markersize=10) for c in colors.values()]
        ax.legend(lines, labels, loc="upper center",
                  bbox_to_anchor=(0.5, -0.1), fancybox=True,
                  shadow=True, ncol=6)

    if "title" in kwargs:
        ax.set_title(kwargs["title"])
    if "xlabel" in kwargs:
        ax.set_xlabel(kwargs["xlabel"])
    if "ylabel" in kwargs:
        ax.set_ylabel(kwargs["ylabel"])

    plt.tight_layout()
    return None if use_ax else fig


def sci_not(num):
    if num == 0:
        return 0, 0
    expon = int(np.floor(np.log10(abs(num))))
    return num*10**(-expon), expon


def diode_analyse(input_file, i, j, DCR=0, sc_print=0):
    path_rp = join(dirname(sys.argv[0]), config['Paths']["path_reports"])
    line = ("+"+69*"-"+"+")
    diode = "PAT19D268"
    spec = "_PDE_" + str(j) + "%_HO_50us_t_30s"

    df = pd.read_csv(input_file, sep=";")
    df["Histo01"] = df["Histo01"]/max(df["Histo01"])
    df["rolling"] = (df["Histo01"].rolling(int(400/13), center=True)
                                  .apply(integrate.trapz, raw=True))

    x0 = df["BinCenter"][df["rolling"].idxmax()]

    df_range = df[(df["BinCenter"] > x0-1000) &
                  (df["BinCenter"] < x0+1000)].copy().reset_index(drop=True)
    df_range["BinCenter"] -= x0

    guess = [
        max(df_range["Histo01"]),
        df_range["BinCenter"][df_range["Histo01"].idxmax()],
        50
        ]
    param, cov = optimization.curve_fit(gaussian, df_range["BinCenter"],
                                        df_range["Histo01"], guess)

    idx1 = df_range["BinCenter"][
        (df_range["BinCenter"] < df_range["BinCenter"][(df_range["Histo01"]
                                                        .idxmax())]) &
        (df_range["BinCenter"] > (df_range["BinCenter"][(df_range["Histo01"]
                                                        .idxmax())]
                                  - 1.5*np.sqrt(2*np.log(2))*param[2])) &
        (df_range["Histo01"] > 0.5)].idxmin()
    xFM1, yFM1 = df_range[["BinCenter", "Histo01"]].iloc[idx1]
    xFM2, yFM2 = df_range[["BinCenter", "Histo01"]].iloc[idx1-1]
    xFMm1, _ = intersection((xFM1, yFM1), (xFM2, yFM2), y=0.5)

    idx2 = df_range["BinCenter"][
        (df_range["BinCenter"] > df_range["BinCenter"][(df_range["Histo01"]
                                                        .idxmax())]) &
        (df_range["BinCenter"] < (df_range["BinCenter"][(df_range["Histo01"]
                                                        .idxmax())]
                                  + 1.5*np.sqrt(2*np.log(2))*param[2])) &
        (df_range["Histo01"] > 0.5)].idxmax()
    xFM3, yFM3 = df_range[["BinCenter", "Histo01"]].iloc[idx2]
    xFM4, yFM4 = df_range[["BinCenter", "Histo01"]].iloc[idx2+1]
    xFMm2, _ = intersection((xFM3, yFM3), (xFM4, yFM4), y=0.5)

    df_range["cumtrapz"] = np.append(0, integrate.cumtrapz(df_range["Histo01"],
                                     df_range["BinCenter"]))
    df_range["cumtrapz"] = df_range["cumtrapz"]/max(df_range["cumtrapz"])

    y_middle = df_range.iloc[-1]["cumtrapz"]/2
    xm1, ym1 = df_range[["BinCenter", "cumtrapz"]][df_range["cumtrapz"]
                                                   < y_middle].max()
    xm2, ym2 = df_range[["BinCenter", "cumtrapz"]][df_range["cumtrapz"]
                                                   > y_middle].min()
    x_middle, _ = intersection((xm1, ym1), (xm2, ym2), x=0.5)

    x1, y1 = df_range[["BinCenter", "cumtrapz"]][df_range["BinCenter"]
                                                 < -600].max()
    x2, y2 = df_range[["BinCenter", "cumtrapz"]][df_range["BinCenter"]
                                                 > -600].min()
    _, area_l = intersection((x1, y1), (x2, y2), x=-600)

    df_range["reverse_cumtrapz"] = (max(df_range["cumtrapz"])
                                    - df_range["cumtrapz"])

    x3 = df_range["BinCenter"][df_range["BinCenter"] < 600].max()
    x4 = df_range["BinCenter"][df_range["BinCenter"] > 600].min()
    y3 = df_range["reverse_cumtrapz"][df_range["BinCenter"] < 600].min()
    y4 = df_range["reverse_cumtrapz"][df_range["BinCenter"] > 600].max()
    _, area_r = intersection((x3, y3), (x4, y4), x=600)

    x5, y5 = df_range[["BinCenter", "cumtrapz"]][df_range["BinCenter"]
                                                 < -200].max()
    x6, y6 = df_range[["BinCenter", "cumtrapz"]][df_range["BinCenter"]
                                                 > -200].min()
    _, area_l2 = intersection((x5, y5), (x6, y6), x=-200)

    df_range["reverse_cumtrapz"] = (max(df_range["cumtrapz"])
                                    - df_range["cumtrapz"])

    x7, y7 = df_range[["BinCenter", "cumtrapz"]][df_range["BinCenter"]
                                                 < 200].max()
    x8, y8 = df_range[["BinCenter", "cumtrapz"]][df_range["BinCenter"]
                                                 > 200].min()
    _, area_r2 = intersection((x7, y7), (x8, y8), x=200)

    FWHM = xFMm2 - xFMm1
    OBA = area_l + area_r
    CA = area_r2 - area_l2
    QBER = OBA/(OBA + CA)

    filename = join(path_rp, "report_"+str(i)+"-"+str(j)+".txt")
    f = open(filename, "w")
    print(line, file=f)
    print("| %-67s |" % "FULL DATA:", file=f)
    print(line, file=f)
    print("  %-45s %s" % ("Diode:", diode), file=f)
    print("  %-45s %s" % ("Spec:", spec[1:]), file=f)
    print("  %-45s %d ps" % ("Rolling integral center x0:", x0), file=f)
    print("  %-45s %d ps" % ("FWHM main pic:", FWHM), file=f)
    print(line, file=f)
    print("| %-67s |" % "NEW SELECTION: [x0-1000ps; x0+1000ps]", file=f)
    print(line, file=f)
    print("  %-45s %1.4f %%" %
          ("Cumulative integral [-1000ps; -600ps]:", 100*area_l), file=f)
    print("  %-45s %1.4f %%" %
          ("Cumulative integral [ 600ps;  1000ps]:", 100*area_r), file=f)
    print("  %-45s %1.4f %%" %
          ("Total Out of Bounds Area:", 100*OBA), file=f)
    print("  %-45s %1.3f %%" %
          ("Central Area [-200ps;  200ps]:", 100*CA), file=f)
    print("  %-45s %1.4f %%" % ("QBER Area Ratio:", 100*QBER), file=f)
    f.close()
    if sc_print:
        with open(filename) as f:
            print(*f.readlines()[1:], sep="")

    # A4 landscape: 11.69, 8.27
    plt.rcParams.update({"font.size": 12})
    f, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(11.69, 8.27),
                                 gridspec_kw={"height_ratios": [3, 1]})
    f.subplots_adjust(top=0.91, left=0.1, right=0.94, hspace=0.16)

    ax1.hlines(0.5, xFMm1, xFMm2, linestyles="solid", color="g", linewidth=1.5)
    if DCR != 0:
        ax1.text(-985, 0.975, "DCR: %1.1f kHz" % (DCR/1000), va="top",
                 ha="left", fontsize=9.5, color="white", weight="bold",
                 bbox=dict(boxstyle="round", facecolor="orange", alpha=0.7))
    ax1.text(xFMm1-15, 0.52, "FWHM: %d ps" % FWHM, va="bottom",
             ha="right", fontsize=9.5, color="white", weight="bold",
             bbox=dict(boxstyle="round", facecolor="green", alpha=0.5))
    ax1.plot(df_range["BinCenter"], df_range["Histo01"],
             color="g", label="Data")
    ax1.plot(df_range["BinCenter"], df_range["cumtrapz"], color="royalblue",
             linestyle="--", label="Cumulative integral")
    ax1.plot(df_range["BinCenter"], df_range["reverse_cumtrapz"],
             color="royalblue", linestyle="--", label="")
    ax1.axhline(y=0.5, color="red", linestyle="-", linewidth=0.5)
    ax1.axvline(x=0, color="red", linestyle="-", linewidth=0.5)
    ax1.axvline(x=x_middle, color="royalblue", linestyle="-", linewidth=0.7)
    ax1.axvspan(-600, -200, color="red", alpha=0.05)
    ax1.axvspan(200, 600, color="red", alpha=0.05)

    ax1.plot(-200, area_l2, "x", color="red")
    ax1.plot(200, area_r2, "x", color="red")
    ax1.text(-200+20, area_l2, "%1.1f %%" % (100*area_l2),
             va="top", ha="left", fontsize=9.5,
             color="white", weight="bold",
             bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))
    ax1.text(200+20, area_r2, "%1.1f %%" % (100*area_r2),
             va="top", ha="left", fontsize=9.5,
             color="white", weight="bold",
             bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))

    ax1.grid(b=True, which="major", linestyle="-")
    ax1.grid(b=True, which="minor", linestyle="--")
    ax1.minorticks_on()
    ax1.legend(loc=5)
    ax1.set_ylabel("Normalized profiles", labelpad=10)
    ax1.yaxis.set_label_coords(-0.055, 0.5)
    ax1.set_title(r"%s-%04d $\cdot$ eff %d%%" % (diode, i, j),
                  fontsize="x-large")
    ax1.set_xlim((min(df_range["BinCenter"]) - 25,
                  max(df_range["BinCenter"]) + 25))
    ax1.set_ylim((-0.025, 1.025))

    ax2.plot(df_range["BinCenter"], 100*df_range["cumtrapz"],
             color="royalblue", linestyle="--")
    ax2.plot(df_range["BinCenter"], 100*df_range["reverse_cumtrapz"],
             color="royalblue", linestyle="--")

    ax2.axvspan(-600, -200, color="red", alpha=0.05)
    ax2.axvspan(200, 600, color="red", alpha=0.05)
    ax2.plot(-600, 100*area_l, "x", color="red")
    ax2.plot(600, 100*area_r, "x", color="red")
    ax2.axvline(x=0, color="red", linestyle="-", linewidth=0.5)
    ax2.axvline(x=x_middle, color="royalblue", linestyle="-", linewidth=0.7)

    ax2.text(-600-20, 100*(area_l+0.1*area_r), "%1.3f %%" % (100*area_l),
             va="bottom", ha="right", fontsize=9.5,
             color="white", weight="bold",
             bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))
    ax2.text(600+20, 1.1*100*area_r, "%1.3f %%" % (100*area_r),
             va="bottom", ha="left", fontsize=9.5,
             color="white", weight="bold",
             bbox=dict(boxstyle="round", facecolor="red", alpha=0.5))

    ax2.grid(b=True, which="major", linestyle="-")
    ax2.grid(b=True, which="minor", linestyle="--")
    ax2.minorticks_on()
    ax2.set_title("QBER: %1.3f x $10^{%d}$" % sci_not(QBER), pad=12)
    ax2.set_ylabel("Cululative integral [%]", labelpad=10)
    ax2.yaxis.set_label_coords(-0.055, 0.5)
    ax2.set_xlabel("Relative time [ps]")
    ax2.set_ylim((-0.025, 2*100*max(area_l, area_r)))
    ax2.locator_params(nbins=5, axis="y")
    ax2.locator_params(nbins=20, axis="x")

    filename = join(path_rp, "plot_%d-%d.png" % (i, j))
    plt.savefig(filename, dpi=300)
    plt.show() if sc_print else plt.close()
    plt.rcdefaults()

    print("  :-) %s was analysed" % input_file)
    return FWHM, QBER


def colorFader(c1, c2, mix=0):
    c1 = np.array(mpl.colors.to_rgb(c1))
    c2 = np.array(mpl.colors.to_rgb(c2))
    return mpl.colors.to_hex((1-mix)*c1 + mix*c2)


path_ds = join(dirname(sys.argv[0]), config['Paths']["path_datasets"])
path_rp = join(dirname(sys.argv[0]), config['Paths']["path_reports"])
filename_sumplot = join(path_rp, "summary.png")
filename_sumpickle = join(path_rp, "summary.pickle")
filename_sumcsv = join(path_rp, "summary.csv")

line = "+"+69*"-"+"+"

rex = re.compile(r"^([\s]*[0-9]+[\s]*((-|,)[\s]*[0-9]+[\s]*)*|all|a|)$")

print(line)
print("| %-67s |" % "SELECTION OF DIODES TO ANALYSE")
print(line)
while True:
    n = input("  %-45s " % "IDs (e.g. 2,7-9 or all)")
    eff = input("  %-45s " % "Efficiencies: (e.g. 10)")

    if rex.match(n) and rex.match(eff):
        print(line)
        break
    else:
        print("  /!\\ Oops! You selected a wrong option, retry...")
        print(line)

startTime = datetime.now()
files = sorted([f for f in listdir(path_ds) if isfile(join(path_ds, f))])

if any([*[x in ["a", "all"] for x in [n, eff]],
        not n.strip(), not eff.strip()]):
    neff = []
    for f in files:
        neff.append([int(f.split("_")[1]), int(f.split("_")[3][:-5])])
else:
    eff = list(map(int, eff.replace(" ", "").split(",")))
    n = str2list(n)
    neff = [[x, y] for x in n for y in eff]

neff.sort()

df2 = pd.read_excel(join(dirname(sys.argv[0]), "woriroo_v2.xlsx"))
df2 = df2.rename(columns={"Diode_SN": "ID", "Eff": "eff"})

FA = []
for i, j in neff:
    input_file = "diode_%d_PDE_%d%%.csv" % (i, j)
    filename = join(path_ds, input_file)

    if input_file not in files:
        print("  /!\\ %s does not exist" % filename)
        continue

    try:
        DCR = df2["DCRf_50DT"][(df2["ID"] == i) & (df2["eff"] == j)].values[0]
    except (IndexError, ValueError):
        DCR = 0

    FA.append([i, j, *diode_analyse(filename, i, j, DCR,
                                    1 if len(neff) == 1 else 0)])

if len(FA) == 0:
    print("  --> No valid results found")
    sys.exit()

df = pd.DataFrame(FA, columns=("ID", "eff", "FWHM", "QBER"))
df = df.set_index(['ID', 'eff'])
df2 = df2.set_index(['ID', 'eff'])

df_merge = (pd.merge(df, df2, on=["ID", "eff"], how="left")
            .sort_values(by=["ID", "eff"]))

# Using multiindex for columns: stacking and unstacking commands
# df_merge.columns = pd.MultiIndex.from_arrays([["FWHM", "QBER", "DCR", "DCR"],
#                                               ['', '', "50DT", "20DT"]])
# df_merge.columns = [col[0]+col[1] if not col[1] else col[0]+"_"+col[1]
#                     for col in df_merge.columns]

fig = circle_plot(ids=df.index.get_level_values("ID"),
                  kind=df.index.get_level_values("eff").astype(str)+"%",
                  x=100*df["QBER"].values, y=df["FWHM"].values/1000,
                  title="Summary plot of selected diodes",
                  xlabel="QBER [%]",
                  ylabel="FWHM [ns]")

x = np.linspace(*plt.gca().get_xlim(), 50)
y = np.linspace(*plt.gca().get_ylim(), len(x))

for i in range(len(x)-1):
    plt.axvspan(x[i], x[i+1], alpha=0.1,
                facecolor=colorFader("yellow", "red",
                                     (x[i]-x[0])/(x[-2]-x[0])))
    plt.axhspan(y[i], y[i+1], alpha=0.1,
                facecolor=colorFader("yellow", "red",
                                     (x[i]-x[0])/(x[-2]-x[0])))

print(line)
print("  %-29s %d" % ("Number of analysed diodes:", len(FA)))
print("  %-24s %-4s %s" % ("Selection criterium:", "n",
                           list2str([row[0] for row in neff])))
print("  %28s  %s" % ("eff", list2str([row[1] for row in neff])))
print("  %-29s %-4d +/- %-4d ps" % ("The average FWHM is:", df["FWHM"].mean(),
                                    np.nan_to_num(df["FWHM"].std())))
print("  %-29s %4.2f +/- %4.2f  %%" % ("The average QBER is:",
                                       100*df["QBER"].mean(),
                                       100*np.nan_to_num(df["QBER"].std())))
print(line)

fig.savefig(filename_sumplot, dpi=300)
df_merge.to_csv(filename_sumcsv, index=["ID", "eff"], header=True)
with open(filename_sumpickle, "wb") as f:
    pickle.dump(fig, f)
# with open("reports/summary.pickle", "rb") as f: pickle.load(f).show()

print("  %-29s %.7s" % ("Execution time:", datetime.now() - startTime))
plt.show()
