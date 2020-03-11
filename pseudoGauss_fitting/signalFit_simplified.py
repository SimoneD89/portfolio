import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimization
from scipy import special
import pandas as pd


# Definisco il modello Gaussiano, Skew Gaussian e Modified Gaussian
def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


def gaussian_CDF(x, x0, sigma, alpha):
    return 1/2*(1+special.erf(alpha*(x-x0)/sigma/np.sqrt(2)))


def gaussian_skew(x, a, x0, sigma, alpha):
    return gaussian(x, a, x0, sigma)*gaussian_CDF(x, x0, sigma, alpha)


def gaussian_mod(x, a, x0, sigma, lamb):
    return (a*np.exp(lamb/2*(2*x0+lamb*sigma**2-2*x)) *
            special.erfc((x0+lamb*sigma**2-x)/(np.sqrt(2)*sigma)))


# Leggo il dataset specificando il separatore
df = pd.read_csv("PAT19D268-0013_PDE_10%_HO_25us_t_30s.csv", sep=";")

# Ascissa x="BinCenter" e ordinata y="Hist01" del dataset
x = df.loc[:, "BinCenter"]/1000  # Tempo in nanosecondi
y = df.loc[:, "Histo01"]  # Counts

# Fit dei dati col modello gaussiano
guess = [max(y), x[y.idxmax()], 1]  # Guess dei parametri
param_gauss, _ = optimization.curve_fit(gaussian, x, y, guess)

# Fit dei dati col modello Modified Gaussian
guess = [max(y), x[y.idxmax()], 1, 0]
param_mod, _ = optimization.curve_fit(gaussian_mod, x, y, guess)

# Fit dei dati col modello Skew Gaussian
guess = [max(y), x[y.idxmax()], 1, 1]
param_skew, _ = optimization.curve_fit(gaussian_skew, x, y, guess)

# Inizializzo la figura
plt.figure()
plt.rcParams.update(plt.rcParamsDefault)

# Disegno i punti (x, y) del dataset
plt.plot(x, y, "o", color="red", markersize=3.2, alpha=0.5, label="Data")
# Disegno la curva Gaussiana che e' il risultato della prima regressione
plt.plot(x, gaussian(x, *param_gauss), "--", color="orange", linewidth=1.5,
         alpha=0.8, label="Gaussian Model")
# Disegno la curva Modified Gaussian, risultato della seconda regressione
plt.plot(x, gaussian_mod(x, *param_mod), "--", color="green", linewidth=1.5,
         alpha=1, label="Modified Model")
# Disegno la curva Skew Gaussian che e' il risultato della seconda regressione
plt.plot(x, gaussian_skew(x, *param_skew), "--", color="blue", linewidth=1.5,
         alpha=0.8, label="Skew Model")

# Setto alcuni parametri del plot come il titolo e il label degli assi
plt.title("Different regression models")
plt.ylabel("Counts")
plt.xlabel("Time [ns]")
plt.legend()
plt.grid(b=True, which="major", linestyle="-")
plt.grid(b=True, which="minor", linestyle="--")
plt.minorticks_on()
plt.xlim((74.57, 75.5))

# Salvo la figura e la stampo a schermo. I dpi corrispondono alla qualità
plt.savefig("signalFit_simp.png", dpi=400)
plt.show()
