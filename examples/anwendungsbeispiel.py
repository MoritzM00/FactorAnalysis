"""
This python file contains all cells of the `anwendungsbeispiel.ipynb` notebook.
"""


from os import getcwd
from pathlib import Path
from warnings import filterwarnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from factor_analyzer.factor_analyzer import calculate_kmo
from sklearn.neighbors import LocalOutlierFactor

from factor_analysis import FactorAnalysis
from factor_analysis.plotting import create_loadings_heatmaps, scree_plot

filterwarnings("ignore")
np.set_printoptions(suppress=True, precision=8)
pd.set_option("display.precision", 4)


# # Anwendungsbeispiel: California Housing Data (1990)
#
# In der Datensatzbeschreibung heißt es:
# > This dataset appeared in a 1997 paper titled Sparse Spatial Autoregressions by Pace, R. Kelley and Ronald Barry, published in the Statistics and Probability Letters journal.
# > They built it using the 1990 California census data.
# >
# > It contains one row per census block group. A block group is the smallest geographical unit for which the U.S. Census Bureau publishes sample data
# > (a block group typically has a population of 600 to 3,000 people).
#
# Das bedeutet, dass jede Beobachtung eine sogenannte [*block group*](https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4) angibt. Eine 'Block-Gruppe'
# ist eine statistische Aufteilung von des Volkszählungsamtes der USA, die zwischen 600 und 3000 Menschen umfassen sollten.
#
# Anstelle von Block-Gruppe werden wir hier meistens vereinfachend den Begriff Bezirk oder Gegend benutzen.

# ## Beschreibung der Daten

# In[ ]:


path = Path(getcwd(), "data", "cal_housing.data")
X = pd.read_csv(path, header=0, sep=",")
X.info()


# Insgesamt haben wir 20640 Beobachtungen (Bezirke) und 9 metrisch skalierte Merkmale. Also gibt es auch keine fehlende Werte,
# wie man im Output der `info` Methode sehen können.
#
# Für die Faktoranalyse benötigen wir metrisch skalierte Merkmale. Vorliegend sind alle neun Merkmale metrisch, weshalb wir in diesem
# Bereich keine Probleme haben.

# In[ ]:


X.describe()


# Die Merkmale lassen sich also wie folgt beschreiben:
#
# - `longitude`: Längengrad eines Bezirks, Werte zwischen -124.35 und -114.31
# - `latitude`: Breitengrad eines Bezirks, Werte zwischen -32.54 und 41.95
# - `housing_median_age`: Median des Alters der Häuser in einem Bezirk, Werte zwischen 1 und 52 (Jahre)
# - `total_rooms`: Gesamtzahl der Räume eines Bezirks, Werte zwischen 2 und 39320
# - `total_bedrooms`: Gesamtzahl der Schlafzimmer eines Bezirks, Werte zwischen 1 und 6445
# - `population`: Einwohnerzahl eines Bezirks, Werte zwischen 3 und 35682
# - `households`: Gesamtzahl der Haushälte eines Bezirks, Werte zwischen 1 und 6082
# - `median_income`: Median des Einkommens der Einwohner, Werte zwischen 0.5 und 15
# - `median_house_value`: Median des Geldwertes der Häuser, Werte zwischen 15000 und 500 001 Dollar.
#
# Dabei fällt besonders der Wertebereich von `median_income` auf. Dies ist wahrscheinlich eine angepasste Skala und kein Einkommen in Dollar.
#
# Wenn wir noch einmal die Beschreibung eines Bezirks anschauen, dann sehen wir, dass in diesem Datensatz
# einige Ausreißer im Hinblick auf `population` vorliegen könnten. Denn ein Bezirk sollte hier zwischen 600 und 3000 Menschen umfassen.

# ### Fehlende Werte
#
# Der Datensatz enthält keine fehlenden Werte, wie wir in der `pd.info` Methode sehen können.

# ### Die Verteilung der Daten
#
# Nun schauen wir uns einige univariate und bivariate Plots der metrischen Merkmale an, um ein Gefühl für die Daten zu bekommen.

# In[ ]:


X.hist(figsize=(20, 15), bins=30)
plt.show()


# Hier sehen wir, dass die vier Merkmale `total_rooms`, `total_bedrooms`, `population` und `households` eine relativ hohe
# Schiefe aufweisen.
# Ebenso hat `median_house_value` bei circa 500 000 eine hohe Dichte. Dies könnte dadurch begründet sein,
# dass der Wert 500 000 als obere Grenze benutzt wurde.
#
# Bevor wir uns also bivariate Plots ansehen, werden wir mittels LocalOutlierFactor (LOF) versuchen, einige Ausreißer zu eliminieren.

# In[ ]:


labels = LocalOutlierFactor(n_neighbors=35).fit_predict(X)
outliers = X[labels == -1]
X = X[labels == 1]

print(
    f"Anzahl an mittels LOF als Ausreißer identifizierte Bezirke: {outliers.shape[0]}"
)


# In[ ]:


X.describe()


# Wir können, sehen, dass der maximale Wert von `population` nun deutlich niedriger ist. Dennoch gibt es Bezirke mit sehr geringen Einwohnerzahlen (Minimum 3 Einwohner).
#
# Schauen wir uns nun einige bivariate Plots genauer an:

# In[ ]:


sns.pairplot(X, diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3})
plt.show()


# In[ ]:


group1 = ["longitude", "latitude"]
group2 = ["total_rooms", "total_bedrooms", "population", "households"]
group3 = ["median_income", "median_house_value"]


# In[ ]:


sns.pairplot(
    X[group1], diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3}, size=5
)
plt.show()


# In[ ]:


sns.pairplot(
    X[group2], diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3}, size=3
)
plt.show()


# In[ ]:


sns.pairplot(
    X[group3], diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3}, size=5
)
plt.show()


# In diesem Pairplot sind univariate Plots auf der Hauptdiagonalen und bivariate Plots (hier Scatter-Plots) auf den nicht-diagonalen Elementen.
# Es fällt auf, dass die Merkmale total_rooms, total_bedrooms, population und households hoch miteinander korrelieren. Dies ist jedoch
# unter Beachtung der Bedeutung der Merkmale nicht überraschend, da eine hohe Anzahl an Menschen im Bezirk schließlich bedeuten muss, dass mehr
# Räume existieren und insgesamt die Zahl der Haushälte wohl höher sein muss.

# In[ ]:


fig, ax = plt.subplots(figsize=(15, 10))
sc = ax.scatter(
    x="longitude",
    y="latitude",
    c="median_house_value",
    cmap="viridis",
    data=X,
    alpha=0.5,
)
plt.colorbar(sc, label="median_house_value")

cities = ["Los Angeles", "San Francisco"]
xys = [(-118.243683, 34.052235), (-122.431297, 37.773972)]
xys_text = [(-121, 33.5), (-123, 35.5)]
for city, xy, xytext in zip(cities, xys, xys_text):
    ax.annotate(
        city,
        xy=xy,
        xycoords="data",
        xytext=xytext,
        arrowprops=dict(facecolor="red", shrink=0.05),
        horizontalalignment="right",
        verticalalignment="top",
    )
plt.tight_layout()
plt.grid()
ax.set_xlabel("Längengrad")
ax.set_ylabel("Breitengrad")
plt.show()


# Die Bezirke mit sehr teuren Häusern liegen am unteren Rand, also an der Küste. Zwei große Städte sind im Bild mit einem Pfeil markiert.

# # Faktoranalyse Schritt 1: Geeignetheit der Daten untersuchen
#
# Bevor wir mit der eigentlichen Faktoranalyse starten, müssen wir die Geeignetheit der Daten überprüfen.
# Dafür benutzen wir das Kaiser-Meyer-Olkin-Kriterium (KMO-Kriterium) für den kompletten Datensatz und das
# dazu verwandte Measure of sampling adequacy (MSA) für jedes einzelne Merkmal.
#
# Generell wollen wir, dass der KMO-Wert über 0.5 ist und der MSA-Wert für jede Variable ebenfalls 0.5 nicht
# unterschreitet.

# In[ ]:


msa_values, kmo = calculate_kmo(X)
print(f"Der KMO-Wert beträgt {kmo:.4f}\n")
msa_df = pd.DataFrame(msa_values.reshape(-1, 1), index=X.columns, columns=["MSA"])
print(msa_df)


# Der KMO-Wert ist über 0.6, was auf eine akzeptable Qualität hinweist. Dies ist jedoch kein idealer Wert.
# Außerdem ist der MSA-Wert für 4 Variablen unter 0.5, jedoch ist der Wert für die anderen Merkmale gut.
#
# Jetzt könnte man beispielweise die Merkmale mit einem MSA-Wert unter 0.5 entfernen. Für dieses Beispiel
# werden wir allerdings mit allen Variablen fortfahren.
#
# Man sollte sich zudem noch die Korrelationsmatrix direkt anschauen. Dies tun wir im Folgenden mit einer
# Heatmap, da sie sich sehr gut als visuelle Repräsentation eignet.

# In[ ]:


fig = plt.figure(figsize=(10, 10))
corr = X.corr().round(2)
hm = sns.heatmap(data=corr, vmax=1, vmin=-1, cmap="RdBu_r", annot=True)
hm.set_xticklabels(X.columns, rotation=45)
hm.set_yticklabels(X.columns)
plt.tight_layout()
plt.show()


# Hier sieht man, dass es mehrere Gruppen von hoch korrelierten Merkmalen gibt. Wie wir schon im Pairplot gesehen haben,
# sind das die vier Merkmale in der Mitte, sowie `housing_median_age` mit einer leicht negativen Korrelation zu diesen vier Merkmalen.
# Ebenso sind `Longitude` und `Latitude` negativ miteinander korreliert. Die Merkmale `median_income` und `median_house_value` sind positiv miteinander korreliert.
#
# Dies deutet schon darauf hin, dass eine 3-Faktorlösung wahrscheinlich eine gute Wahl von k sein könnte. Im Folgenden werden dies genauer betrachten.

# # Faktoranalyse Schritt 2: Wahl der Faktorzahl $k$
#
# Um diese Frage zu beantworten, werden wir zunächst alle Faktoren mit der Hauptkomponentenmethode (Principal Component, PC)
# extrahieren und die Eigenwerte der Faktoren mithilfe eines 'Scree Plots' betrachten.
#
# Dann können wir die Faktoren behalten, die einen Eigenwert größer Eins (Kaiser-Kriterium) haben, oder anhand eines 'Knicks'
# im Plot eine geeignete Faktorzahl erkennen.

# In[ ]:


fa = FactorAnalysis(n_factors=X.shape[1], method="pc").fit(X)


# In[ ]:


fig, ax = plt.subplots(figsize=(13, 8))
scree_plot(fa.eigenvalues_, ax)
ax.set_xlabel("Faktor")
ax.set_ylabel("Eigenwert")
plt.tight_layout()
plt.grid()
plt.show()


# Hier können wir sehen, dass die ersten drei Faktoren einen Eigenwert
# größer als Eins haben. Dies macht im Hinblick auf die Korrelationsmatrix,
# die wir vorhin gesehen haben, wegen den drei 'Boxen' auch Sinn. Nach
# dem Kaiser-Kriterium sollten wir also ein 3-Faktor-Modell benutzen.
#
# Der vierte Faktor ist jedoch nur minimal unter dem Eigenwert Eins, weshalb
# man diesen auch nicht direkt ausschließen sollte. Ein 'Knick' wäre bei
# $k = 5$ erkennbar.
# Wir werden also $k \in [3, 4, 5]$ ausprobieren und den 'Besten' auswählen.

# # Faktoranalyse Schritt 3: Extrahieren der Faktoren
# Jetzt führen wir die eigentliche Faktorextraktion durch.
#
# Dafür werden wir drei unterschiedliche Extraktionsmethoden miteinander vergleichen:
#  - Hauptkomponentenmethode (engl. Principal Components (PC) Method)
#  - Hauptachsen-Faktorisierung (engl. Principal Axis Factoring (PAF))
#  - Iterierte Hauptachsen-Faktorisierung (engl. Iterated Principal Axis Factoring (Iterated PAF))
#
# Dabei ist die letzte Variante wohl die am häufigsten eingesetzte Methode (unter diesen drei).

# In[ ]:


methods = [
    ("PC", FactorAnalysis(method="pc")),
    ("Nicht-iterierte PAF", FactorAnalysis(method="paf", max_iter=1)),
    ("Iterierte PAF", FactorAnalysis(method="paf", max_iter=50)),
]
for n_factors in range(3, 6):
    figsize = (10 + (1 + n_factors) // 2, 8)
    # this convenience method accepts unfitted factor analysis instances
    # and fits them for us. With the fa_params dict we can easily specify
    # the arguments which are shared across all instances.
    create_loadings_heatmaps(X, methods, figsize, fa_params={"n_factors": n_factors})
    plt.gcf().suptitle(f"Ladungsmatrizen eines {n_factors}-Faktor-Modells")
    plt.show()


# Hier sehen wir in jeder Zeile die Ladungsmatrizen der drei unterschiedlichen Methoden als Heatmap dargestellt.
# Wir stellen fest, dass die 3-Faktorlösung tatsächlich im Hinblick auf eine einfache Struktur eine gute Lösung darstellt.
# Die 4-Faktorlösung könnte jedoch auch noch eine valide Lösung sein. Lediglich die 5-Faktorlösung ist problematisch,
# da kein Merkmal hoch (absoluter Wert größer 0.4) auf den fünften Faktor lädt.
#
# Ein Problem ist jedoch, dass das Merkmal `housing_median_age` nur bei der PC-Methode sehr hoch auf den vierten Faktor lädt
# und bei einer 3-Faktorlösung nur eine moderate Ladung auf den ersten Faktor hat.
# Dies deutet auf eine hohe spezifische Varianz hin, d.h. die Faktoren sind nicht gut in der Lage, die Varianz dieses Merkmals
# zu erklären.
#
# Wir werden also die 3-Faktorlösung genauer betrachten.

# In[ ]:


fa_params = {"n_factors": 3}

axes = create_loadings_heatmaps(
    X, methods, figsize=(10, 9), fa_params=fa_params, annotate=True
)
plt.tight_layout()
plt.show()


# Wir können sehen, dass bei der PC-Methode die Ladungen generell höher ausfallen und dass der zweite Faktor ein unterschiedliches
# Vorzeichen besitzt, im Gegensatz zu den anderen beiden Methoden.
#
# Die iterierte und nicht-iterierte PAF-Methode sind sehr ähnlich zueinander. Jedoch ist die iterierte
# Variante oft in Hinblick auf die reproduzierte Korrelationsmatrix besser. Dies können wir anhand des
# *root mean squared error* (RMSE) untersuchen:

# In[ ]:


for method, fa in methods:
    print(f"RMSE von {method}: {fa.get_rmse():.4f}")
    rmse = fa.get_rmse()


# Der root mean squared error of residuals (RMSE) ist bei der iterierten PAF-Methode am geringsten,
# gefolgt von der nicht-iterativen Variante und der Hauptkomponentenmethode.

# Da das Merkmal `housing_median_age` eine sehr hohe spezifische Varianz (geringe Kommunalität) aufweist
# können wir auch ein 3-Faktor-Modell ohne diesem Merkmal anschauen.
# Dieses kann den RMSE um knapp 44% reduzieren (im Vergleich sind die iterierten PAF-Methoden)

# In[ ]:


X_without_age = X.drop(columns="housing_median_age", axis=1)
fa_without_age = FactorAnalysis(n_factors=3).fit(X_without_age)
perc = 1 - fa_without_age.get_rmse() / rmse
print(f"Der RMSE konnte durch entfernen des Merkmals um {perc:.2%} reduziert werden")


#
# Den Unterschied zwischen den Methoden im RMSE können wir auch noch grafisch analysieren:

# In[ ]:


fig, axes = plt.subplots(ncols=2, figsize=(12, 6))
fitted_methods = [
    ("PC", FactorAnalysis(method="pc", n_factors=3).fit(X)),
    ("Iterierte PAF", FactorAnalysis(method="paf", n_factors=3, max_iter=50).fit(X)),
]
for ax, (method, fa) in zip(axes, fitted_methods):
    R = fa.corr_
    R_hat = fa.get_reprod_corr()
    abs_residuals = np.abs(R - R_hat)
    mask = np.triu(np.ones_like(R))
    ax.set_title(f"{method} (RMSE = {fa.get_rmse():.4f})", fontsize=11)
    s = sns.heatmap(
        abs_residuals.round(2),
        cmap="BuGn",
        ax=ax,
        cbar=False,
        annot=True,
        square=True,
        mask=mask,
    )
    s.set_xticklabels(range(1, 10))
    s.set_yticklabels(range(1, 10), rotation=0)
fig.suptitle("Residualmatrizen von zwei Extraktionsmethoden mit k=3")
fig.tight_layout()
plt.show()


# In[ ]:


# Zusammenfassung der iterierten PAF-Methode
methods[2][1].print_summary()


# In der Zusammenfassung können wir neben den Ladungen auch die Kommunalitäten und spezifischen Varianzen,
#  sowie Eigenwerte und den Anteil erklärter Varianz durch die Faktoren betrachten .
#
#
# Wir können sehen, dass die spezifischen Varianze von 'housing_median_age' mit einem Wert zwischen 0.90 sehr hoch ist.
# Das bedeutet, dass die Faktoren die Varianz dieses Merkmals gemeinsam nicht besonders gut erklären können. Dies spiegelt sich ebenfalls in den
# sehr geringen Ladungen auf die drei Faktoren wider.
# Die spezifischen Varianzen bei den restlichen Merkmalen sind jedoch sehr niedrig,
# was ein gutes Zeichen für die Qualität der Faktorlösung ist.
#
# Alle Merkmale, bis auf des eben angesprochenen Merkmals können eindeutig einem Faktor durch jeweils die betraglich
# größte Ladung zugeordnet werden. Dies ist nicht immer der Fall, sodass eine Faktorrotation für eine leichtere Interpretation
# sorgen könnte. Hier ist jedoch auch ohne Rotation eine Interpretation gut möglich. Wir werden im nächsten Schritt trotzdem
# beispielhaft die mit der Varimax-Methode rotierten Faktorladungen ansehen.
#
# Bevor wir das tun, werden wir noch die verschiedenen initialen Schätzungen der Kommunalitäten in der (iterierten) PAF-Methode vergleichen.
# Interessant könnte dabei sein, ob die Wahl der initialen Schätzung einen Einfluss auf die finalen Kommunalitäten hat.

# In[ ]:


paf_comparison_methods = [
    ("Nicht-iterierte PAF", FactorAnalysis(method="paf", max_iter=1)),
    ("PAF mit max_iter=3", FactorAnalysis(method="paf", max_iter=3)),
    ("Iterierte PAF", FactorAnalysis(method="paf", max_iter=50)),
]
figsize = (8, 6)
initial_communality_estimates = {
    "smc": "Quadrierte multiple Korrelationen (SMC)",
    "mac": "Maximale absolute Korrelationen (MAC)",
    "ones": "Einsen",
}
for init_comm in initial_communality_estimates:
    print(f"Initiale Schätzung: {initial_communality_estimates[init_comm]}")
    create_loadings_heatmaps(
        X,
        paf_comparison_methods,
        figsize,
        fa_params={"n_factors": 3, "initial_comm": init_comm},
    )
    plt.show()
    print(
        f"Iterierte PAF hat {paf_comparison_methods[2][1].n_iter_} Iterationen benötigt."
    )
    for method, fa in paf_comparison_methods:
        print(f"RMSE von {method}: {fa.get_rmse():.4f}")
    print("\n")


# Wir können sehen, dass nur geringe Unterschiede zwischen den unterschiedlichen initialen Schätzungen
# in den Ladungen feststellbar sind. Nur in der nicht-iterierten Variante der PAF-Methode können wir einige
# Unterschiede, vor allem im zweiten Faktor feststellen. Beispielsweise hat der zweite Faktor ein unterschiedliches
# Vorzeichen, jedoch nur wenn Einsen als Kommunalitätsschätzung benutzt werden.
#
# Wir stellen fest, dass die iterierte Variante jedoch eine unterschiedliche Anzahl an Iterationen benötigt, bis
# das Konvergenzkriterium erreicht wird. Am Langsamsten ist es mm Falle von Einsen (10 Iterationen) und am schnellsten ist es
# bei den maximalen absoluten Korrelationen (MAC) (nur 3 Iterationen). Ist das Konvergenzkriterium erfüllt, sind die Ladungen jedoch
# bei allen drei initialen Schätzungen weitestgehend identisch.

# # Faktoranalyse Schritt 4: Faktorrotation und -interpretation
# Jetzt rotieren wir die Ladungen mit der Varimax-Methode und versuchen, die Faktoren zu interpretieren.

# In[ ]:


methods = [
    ("Unrotiert", FactorAnalysis(method="paf", rotation=None)),
    ("Varimax-Rotation", FactorAnalysis(method="paf", rotation="varimax")),
]
fa_params = {"n_factors": 3}
fig = create_loadings_heatmaps(X_without_age, methods, fa_params=fa_params)
plt.tight_layout()


# # Faktoranalyse Schritt 5: Die Faktorwerte bestimmen
#
# Als letzten Schritt können wir noch einen Blick auf die geschätzten Faktorwerte
# werfen. Wie hätte als der Bezirk $i$ die drei oben beschriebenen Faktoren bewertet?
#
# Da die Daten standardisiert wurden, sind die Faktorwerte auch (fast) standardisiert.
# Der Mittelwert liegt bei Null, jedoch ist die Varianz der Faktorwerte nicht exakt 1.
# Daher geben die Faktorwerte an, wie weit weg vom Mittelwert ein bestimmter Faktorwert
# $f_{ij}$ des $i$-ten Bezirks für Faktor $j$ liegt.
# Die hier benutzte Methode zur Schätzung der Faktorwerte ist die *Regressionsmethode*
# welche eine multivariate lineare Regression benutzt, um die Faktorwerte
# zu schätzen.

# In[ ]:


scores = FactorAnalysis(n_factors=3).fit_transform(X_without_age)
scores = pd.DataFrame(scores, columns=["Größe", "Standort", "Wohlstand"])
print(scores)


# In[ ]:


scores.std(axis=0) ** 2


# Benutzt man jedoch die Hauptkomponentenmethode, so weisen die Faktorwerte eine
# Standardabweichung von Eins auf.

# In[ ]:


scores = FactorAnalysis(method="pc", n_factors=3).fit_transform(X)
scores.std(axis=0) ** 2
