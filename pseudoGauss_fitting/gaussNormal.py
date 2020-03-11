import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as optimization
import pandas as pd


# Definisco la funzione gaussiana: media x0, deviazione sigma e ampiezza a
def gaussian(x, a, x0, sigma):
    return a*np.exp(-(x-x0)**2/(2*sigma**2))


# Definisco la funzione che ha come input il dataset, il colore della curva e
# l'efficienza che apparira' nella legenda
def gauss_fit(input_file, plot_style, efficiency=0):
    # Leggo il dataset specificando il separatore
    df = pd.read_csv(input_file, sep=";")

    # Ascissa x="BinCenter" e ordinata y="Hist01" del dataset
    x = df.loc[:, "BinCenter"]/1000  # Tempo in nanosecondi
    y = df.loc[:, "Histo01"]  # Counts

    # Fit dei dati col modello gaussiano
    guess = [max(y), x[y.idxmax()], 1]  # Guess dei parametri
    param, cov = optimization.curve_fit(gaussian, x, y, guess)

    # Normalizzo i punti in modo che abbiamo la normalizzazione
    # della gaussiana 1/[sqrt(2pi)|sigma|]
    y = y/param[0]/np.sqrt(2*np.pi)/np.abs(param[2])

    # Disegno i punti (x, y) del dataset
    plt.plot(x, y, "o", color=plot_style, markersize=3, alpha=0.5,
             label="%d%% efficiency" % (efficiency))
    # Disegno la curva gaussiana che e' il risultato della regressione
    plt.plot(np.linspace(min(x), max(x), 10000),
             gaussian(np.linspace(min(x), max(x), 10000),
             1/np.sqrt(2*np.pi)/np.abs(param[2]), *param[1:]), "--",
             color=plot_style, linewidth=1.5, label="")
    # Disegno una linea verticale hlines(a/2, xmin, xmax) a meta' ampiezza
    plt.hlines(0.5/np.sqrt(2*np.pi)/np.abs(param[2]),
               param[1]-np.sqrt(2*np.log(2))*np.abs(param[2]),
               param[1]+np.sqrt(2*np.log(2))*np.abs(param[2]),
               linestyle="dashed", color=plot_style, linewidth=1.5)
    # Inserisco un box sopra la riga a meta' ampiezza che indica la FWHM
    plt.text(param[1], 0.5/np.sqrt(2*np.pi)/np.abs(param[2])+0.15,
             "%d ps" % (1000*2*np.sqrt(2*np.log(2))*np.abs(param[2])),
             horizontalalignment="center", fontsize=9.5, color="white",
             weight="bold", bbox=dict(boxstyle="round", facecolor=plot_style,
                                      alpha=0.6))

    # Ritorno come valori i parametri e la matrice di covarianza del fit
    return param, cov


# Definisco la grandezza della figura e del font
# A4 landscape: 11.69, 8.27
plt.figure(figsize=(11.69, 8.27))
plt.rcParams.update({"font.size": 12})

# Richiamo la funzione che modellizza la curva
gauss_fit("PAT19D268-0013_PDE_10%_HO_25us_t_30s.csv", "royalblue", 10)
gauss_fit("PAT19D268-0013_PDE_15%_HO_25us_t_30s.csv", "green", 15)
gauss_fit("PAT19D268-0013_PDE_20%_HO_25us_t_30s.csv", "red", 20)
gauss_fit("PAT19D268-0013_PDE_25%_HO_25us_t_30s.csv", "orange", 25)

# Setto alcuni parametri del plot come il titolo e il label degli assi
plt.title("Model: Gaussian Distribution")
plt.ylabel("Normalized counts")
plt.xlabel("Time [ns]")
plt.legend(markerscale=2.)
plt.grid(b=True, which="major", linestyle="-")
plt.grid(b=True, which="minor", linestyle="--")
plt.minorticks_on()
plt.xlim((74.57, 75.5))
plt.ylim((-0.3, 9.1))

# Salvo la figura e la stampo a schermo. I dpi corrispondono alla qualità
plt.savefig("FWHMNormalDistribution.png", dpi=400)
plt.show()
plt.rcdefaults()
