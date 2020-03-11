import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.style.use("seaborn-whitegrid")
mpl.rc("text", usetex=True)
mpl.rcParams["text.latex.preamble"] = [
    r"\usepackage{amsmath}",
    r"\usepackage{mathtools}"
]


def theta(n, x):
    if type(x) is int:
        return theta(n, [x])

    y = []
    for i in x:
        if 1 - 1/n < i <= 1:
            y.append(n**2*(i - 1 + 1/n))
        else:
            y.append(0)
    return y


x = np.linspace(0, 1, 1000)

n = [1, 2, 6, 10]
colors = ["royalblue", "green", "orange", "red", "dimgray"]

for i, color in zip(n, colors):
    plt.plot(x, theta(i, x), label=r"$\theta_{%d}(x)$" % i, color=color)
    plt.plot(1, theta(i, 1), "o", mfc="white", color=color)
    plt.plot(1, 0, "o", color=color)
    plt.fill_between(x, 0, theta(i, x), color=color, alpha=0.1)

plt.gca().yaxis.tick_right()
plt.grid(b=True, which="major", linestyle="-")
plt.grid(b=True, which="minor", linestyle="--")
plt.minorticks_on()

plt.text(0.5, 8.8,
         r"\["
         r"\theta_n(x) ="
         r"\begin{cases}"
         r"0, & 0 \leq x \leq 1-\frac{1}{n}\\"
         r"n^2\left(x+\frac{1-n}{n}\right), & 1-\frac{1}{n} \leq x < 1 \\"
         r"0, & x = 1"
         r"\end{cases}"
         r"\]",
         size=10, ha="center", va="top", weight="bold", clip_on=True,
         bbox={"boxstyle": "round", "color": colors[-1], "alpha": 0.1})

plt.xlabel(r"asse x")
plt.ylabel(r"$\theta_n(x)$", rotation=270)
plt.gca().yaxis.set_label_position("right")
plt.title(r"Sequenza di funzioni $\theta_n(x)$ che convergono verso "
          r"la funzione nulla per $n\to\infty$." "\n"
          r"L'area sottesa alla funzione $\theta_n$ è costante e vale "
          r"$\mathcal{A}(\theta_n)\coloneqq\int_0^1\theta_n(x)dx=\frac{1}{2}$")

plt.plot([0.4, 0.45], [4, 4], "--", color=colors[-1])
plt.plot([0.45, 0.55], [4, 4], color=colors[-1])
plt.plot([0.55, 0.6], [4, 6], color=colors[-1])
plt.plot(0.6, 6, "o", mfc="white", markersize=4, color=colors[-1])
plt.plot(0.6, 4, "o", markersize=4, color=colors[-1])
plt.fill_between([0.55, 0.6], 4, [4, 6], color=colors[-1], alpha=0.1)
plt.text(0.61, 4.9, "$n$", color=colors[-1])
plt.text(0.568, 3.4, r"$\frac{1}{n}$", color=colors[-1])
plt.text(0.4, 5, r"$\mathcal{A}=\frac{1}{2}(n\cdot\frac{1}{n})$",
         color=colors[-1])

plt.legend(ncol=4, loc="upper center")

plt.tight_layout()
plt.savefig("theta_n.png", dpi=400)
plt.show()
