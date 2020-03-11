import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimization
import pandas as pd
from scipy import special
from scipy.optimize import fsolve
from functools import partial
from matplotlib.ticker import MaxNLocator


def gaussian(x):
    return np.exp(-x**2/2)


def gaussian_CDF(x):
    return 1/2*(1+special.erf(x/np.sqrt(2)))


def gaussian_skew(x, a, x0, sigma, alpha):
    return a*gaussian((x-x0)/sigma)*gaussian_CDF(alpha*(x-x0)/sigma)


def gaussian_skew_shifted(x, a, x0, sigma, alpha, A):
    return gaussian_skew(x, a, x0, sigma, alpha) - A


def exp(x, A, b, x0):
    return A+np.exp(b*(x-x0))


def find_FW(param, r):
    delta = param[3]/np.sqrt(1+param[3]**2)
    mu = np.sqrt(2/np.pi)*delta
    sigmaz = np.sqrt(1-mu**2)
    gamma = (4-np.pi)/2*(delta*np.sqrt(2/np.pi))**3/(1-2*delta**2/np.pi)**(3/2)

    mean = param[1] + param[2]*delta*np.sqrt(2/np.pi)
    variance = param[2]**2*(1-2*delta**2/np.pi)
    mode = (param[1] +
            param[2]*(mu-1/2*gamma*sigmaz -
                      np.sign(param[3])/2*np.exp(-2*np.pi/np.abs(param[3]))))
    height = gaussian_skew(mode, 2./np.sqrt(2*np.pi)/np.abs(param[2]),
                           *param[1:])

    xmin = fsolve(partial(gaussian_skew_shifted,
                  a=2/np.sqrt(2*np.pi)/np.abs(param[2]), x0=param[1],
                  sigma=param[2], alpha=param[3], A=height*r),
                  mean-np.sqrt(variance), factor=0.001)[0]
    xmax = fsolve(partial(gaussian_skew_shifted,
                  a=2/np.sqrt(2*np.pi)/np.abs(param[2]), x0=param[1],
                  sigma=param[2], alpha=param[3], A=height*r),
                  mean+np.sqrt(variance), factor=0.001)[0]

    FW = xmax-xmin
    return FW, xmin, xmax, mode, height


def gauss_fit(input_file, ax, plot_style, efficiency):
    df = pd.read_csv(input_file, sep=";")

    x = df.loc[:, "BinCenter"]/1000  # Tempo in nanosecondi
    y = df.loc[:, "Histo01"]

    guess = [max(y), x[y.idxmax()], 1, 1]
    param, cov = optimization.curve_fit(gaussian_skew, x, y, guess)

    param[2] = np.abs(param[2])
    y = y/param[0]*2/np.sqrt(2*np.pi)/np.abs(param[2])

    FW, xmin, xmax, mode, height = find_FW(param, 0.5)

    ax.plot(mode, height, "x", color=plot_style)
    ax.text(xmin+FW/2, height/2, "%d ps" % (1000*FW),
            verticalalignment="center", horizontalalignment="center",
            fontsize=9.5, color="white",  weight="bold",
            bbox=dict(boxstyle="round", facecolor=plot_style, alpha=0.6))
    ax.plot(x, y, "o", color=plot_style, markersize=3, alpha=0.5,
            label="%d%% efficiency" % (efficiency))
    ax.plot(np.linspace(min(x), max(x), 10000),
            gaussian_skew(np.linspace(min(x), max(x), 10000),
            2/np.sqrt(2*np.pi)/np.abs(param[2]), *param[1:]), "--",
            color=plot_style, linewidth=1.5, label="")
    ax.hlines(height/2, xmin, xmax, linestyle="dashed", color=plot_style,
              linewidth=1.5)

    return param, cov


def exp_fit(input_file, ax, gauss_param, plot_style):
    df = pd.read_csv(input_file, sep=";")
    df["BinCenter"] = df["BinCenter"]/1000
    df["Histo01"] = df["Histo01"]/(gauss_param[0]*np.sqrt(2*np.pi) *
                                   np.abs(gauss_param[2]))

    x = df.loc[:, "BinCenter"]
    y = df.loc[:, "Histo01"]

    _, xmin, xmax, mode, height = find_FW(gauss_param, 0.01)

    df1 = df[(df["BinCenter"] > xmin - 6*gauss_param[2]) &
             (df["BinCenter"] < xmin + 0.4*gauss_param[2])]
    df2 = df[(df["BinCenter"] > xmax - 0.5*gauss_param[2]) &
             (df["BinCenter"] < xmax + 5*gauss_param[2])]

    param_exp1, _ = optimization.curve_fit(exp, df1["BinCenter"],
                                           df1["Histo01"], [0, 100, xmin],
                                           maxfev=3000)
    param_exp2, _ = optimization.curve_fit(exp, df2["BinCenter"],
                                           df2["Histo01"], [0, -10, xmax],
                                           maxfev=3000)

    xmin = np.log(height/100 - param_exp1[0])/param_exp1[1] + param_exp1[2]
    xmax = np.log(height/100 - param_exp2[0])/param_exp2[1] + param_exp2[2]
    FW = xmax - xmin

    ax.plot(x, y, "o", color=plot_style, markersize=3, alpha=0.5)
    ax.text(xmin + FW/2, height/100, "%d ps" % (1000*FW),
            verticalalignment="center", horizontalalignment="center",
            fontsize=9.5, color="white",  weight="bold",
            bbox=dict(boxstyle="round", facecolor=plot_style, alpha=0.6))
    ax.plot(np.linspace(min(df1["BinCenter"]), max(df1["BinCenter"]), 10000),
            exp(np.linspace(min(df1["BinCenter"]),
                            max(df1["BinCenter"]),
                            10000
                            ),
            *param_exp1), ":", color=plot_style, linewidth=1.5)
    ax.plot(np.linspace(min(df2["BinCenter"]), max(df2["BinCenter"]), 10000),
            exp(np.linspace(min(df2["BinCenter"]),
                            max(df2["BinCenter"]),
                            10000
                            ),
            *param_exp2), ":", color=plot_style, linewidth=1.5)
    ax.hlines(height/100., xmin, xmax, linestyle="dashed", color=plot_style,
              linewidth=1.5)

    return param_exp1, param_exp2


# A4 landscape: 11.69, 8.27
plt.rcParams.update({"font.size": 12})
f, (ax1, ax2) = plt.subplots(2, 1, figsize=(11.69, 8.27), sharex=True,
                             gridspec_kw={"height_ratios": [6, 3]})

f.subplots_adjust(hspace=0)

param, _ = gauss_fit("PAT19D268-0013_PDE_10%_HO_25us_t_30s.csv", ax1,
                     "royalblue", 10)
param1, _ = gauss_fit("PAT19D268-0013_PDE_15%_HO_25us_t_30s.csv", ax1,
                      "green", 15)
param2, _ = gauss_fit("PAT19D268-0013_PDE_20%_HO_25us_t_30s.csv", ax1,
                      "red", 20)
param3, _ = gauss_fit("PAT19D268-0013_PDE_25%_HO_25us_t_30s.csv", ax1,
                      "orange", 25)

exp_fit("PAT19D268-0013_PDE_10%_HO_25us_t_30s.csv", ax2, param, "royalblue")
exp_fit("PAT19D268-0013_PDE_15%_HO_25us_t_30s.csv", ax2, param1, "green")
exp_fit("PAT19D268-0013_PDE_20%_HO_25us_t_30s.csv", ax2, param2, "red")
exp_fit("PAT19D268-0013_PDE_25%_HO_25us_t_30s.csv", ax2, param3, "orange")

ax1.grid(b=True, which="major", linestyle="-")
ax1.grid(b=True, which="minor", linestyle="--")
ax1.minorticks_on()

ax2.grid(b=True, which="major", linestyle="-")
ax2.grid(b=True, which="minor", linestyle="--")
ax2.minorticks_on()
ax2.set_ylim([0.01, 1.14*0.087])

plt.xlabel("Time [ns]")
ax1.legend(markerscale=2)

plt.xlim((74.57, 75.5))

ax1.xaxis.set_major_locator(MaxNLocator(10))

ax1.set_title("Full Width approximation for diode PAT19D268-0013")
ax1.set_ylabel("FWHM", rotation=270, labelpad=11)
ax2.set_ylabel("FW1%", rotation=270, labelpad=11)
ax1.yaxis.set_label_position("right")
ax2.yaxis.set_label_position("right")

f.text(0.075, 0.5, "Normalized counts", va="center", ha="center",
       rotation="vertical")

plt.savefig("FWHMSkewDistribution_v3.png", dpi=500)
plt.show()
plt.rcdefaults()
