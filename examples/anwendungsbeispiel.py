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
# Jede Beobachtung (Zeile) im Datensatz stellt eine sogenannte [*block group*](https://www.census.gov/programs-surveys/geography/about/glossary.html#par_textimage_4) in Kalifornien dar.
#
# Eine 'Block-Gruppe' ist eine statistische Aufteilung von des Volkszählungsamtes der USA,
# die zwischen 600 und 3000 Menschen umfassen sollten.
# Anstelle von Block-Gruppe werden wir hier meistens vereinfachend den Begriff *Bezirk* benutzen.

# ## Beschreibung der Daten

# In[2]:


path = Path(getcwd(), "data", "cal_housing.data")
X = pd.read_csv(path, header=0, sep=",")
X.info()


# Insgesamt haben wir 20640 Beobachtungen (Bezirke) und 9 metrisch skalierte Merkmale.
#
# Da wir für die Faktoranalyse auch metrisch skalierte Merkmale benötigen, haben wir in diesem
# Bereich keine Probleme.

# In[3]:


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

# In[4]:


X.hist(figsize=(20, 15), bins=30)
plt.show()


# Hier sehen wir, dass die vier Merkmale `total_rooms`, `total_bedrooms`, `population` und `households` eine relativ hohe
# Schiefe aufweisen.
# Ebenso hat `median_house_value` bei circa 500 000 eine hohe Dichte. Dies könnte dadurch begründet sein,
# dass der Wert 500 000 als obere Grenze benutzt wurde.
#
#
# ### Ausreißeranalyse
# Bevor wir uns also bivariate Plots ansehen, werden wir mittels LocalOutlierFactor (LOF) versuchen, einige Ausreißer zu eliminieren.

# In[5]:


labels = LocalOutlierFactor(n_neighbors=35).fit_predict(X)
outliers = X[labels == -1]
X = X[labels == 1]

print(
    f"Anzahl an mittels LOF als Ausreißer identifizierte Bezirke: {outliers.shape[0]}"
)


# In[6]:


X.describe()


# Wir können, sehen, dass der maximale Wert von `population` nun deutlich niedriger ist. Dennoch gibt es Bezirke mit sehr geringen Einwohnerzahlen (Minimum 3 Einwohner).
#
# Schauen wir uns nun die Verteilungen der Merkmale mittels eines Pairplots genauer an.

# In[30]:


sns.pairplot(X, diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3})
plt.show()


# In diesem Pairplot sind univariate Plots auf der Hauptdiagonalen und bivariate Plots auf den nicht-diagonalen Elementen.
# Hier fallen besonders drei Gruppen auf:
#
# ##### Gruppe 1
# Die Merkmale `longitude` und `latitude` weisen eine relativ hohe negative Korrelation auf.
#
#
# ##### Gruppe 2
# Die Merkmale `total_rooms`, `total_bedrooms`, `population` und `households` korrelieren hoch miteinander. Dies ist jedoch
# unter Beachtung der Bedeutung der Merkmale nicht überraschend, da eine hohe Anzahl an Menschen im Bezirk schließlich bedeuten muss, dass mehr
# Räume existieren und insgesamt die Zahl der Haushälte wohl höher sein muss.
#
# Das Merkmal `housing_median_age` könnte auch noch zu dieser Gruppe gezählt werden, da es eine leicht negative Korrelation zu diesen vier Merkmalen aufzeigt.
#
#
# ##### Gruppe 3
# Die Merkmale `median_income` und `median_house_value` weisen ebenfalls eine sichtbare positive Korrelation auf, aber nicht so stark wie die in Gruppe 2.
#
#
# Diese drei Gruppen sind unten noch einmal in gesonderten Pairplots dargestellt.

# In[31]:


group1 = ["longitude", "latitude"]
group2 = ["total_rooms", "total_bedrooms", "population", "households"]
group3 = ["median_income", "median_house_value"]


# In[28]:


sns.pairplot(
    X[group1], diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3}, size=4
)
plt.show()


# In[10]:


sns.pairplot(
    X[group2], diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3}, size=3
)
plt.show()


# In[29]:


sns.pairplot(
    X[group3], diag_kind="kde", kind="scatter", plot_kws={"alpha": 0.3}, size=4
)
plt.show()


# Im letzten Pairplot fällt auch wieder die horizontale (bzw. vertikale) Linie bei `median_house_value = 500 000` auf.

# Zudem können wir uns noch die graphische Verteilung der Häuserpreise ansehen.
# Unten dargestellt ist ein Scatter-Plot, der die Lage der Bezirke in Abhängigkeit
# des Medians der Häuserpreise berücksichtigt.

# In[12]:


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
# Zunächst müssen wir die Geeignetheit der Daten für die Faktoranalyse überprüfen.
# Dafür benutzen wir das Kaiser-Meyer-Olkin-Kriterium (KMO-Kriterium) für den kompletten Datensatz und das
# dazu verwandte Measure of sampling adequacy (MSA) für jedes einzelne Merkmal.
#
# Generell wollen wir, dass der KMO-Wert über 0.5 ist und der MSA-Wert für jede Variable ebenfalls 0.5 nicht
# unterschreitet.

# In[13]:


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

# In[14]:


fig = plt.figure(figsize=(10, 10))
corr = X.corr().round(2)
hm = sns.heatmap(data=corr, vmax=1, vmin=-1, cmap="RdBu_r", annot=True)
hm.set_xticklabels(X.columns, rotation=45)
hm.set_yticklabels(X.columns)
plt.tight_layout()
plt.show()


# Wie wir schon im Pairplot gesehen haben, gibt es drei Gruppen von hoch korrelierten Merkmalen:
# Die vier Merkmale in der Mitte, sowie `housing_median_age` mit einer leicht negativen Korrelation zu diesen vier Merkmalen.
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

# In[15]:


fa = FactorAnalysis(n_factors=X.shape[1], method="pc").fit(X)


# In[16]:


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

# In[17]:


methods = [
    ("PC", FactorAnalysis(method="pc")),
    ("Nicht-iterierte PAF", FactorAnalysis(method="paf", max_iter=1)),
    ("Iterierte PAF", FactorAnalysis(method="paf", max_iter=50)),
]
for n_factors in range(3, 6):
    figsize = (10 + (1 + n_factors) // 2, 8)
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

# In[18]:


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

# In[19]:


for method, fa in methods:
    print(f"RMSE von {method}: {fa.get_rmse():.4f}")
    rmse = fa.get_rmse()


# Der root mean squared error of residuals (RMSE) ist bei der iterierten PAF-Methode am geringsten,
# gefolgt von der nicht-iterativen Variante und der Hauptkomponentenmethode.

# Da das Merkmal `housing_median_age` eine sehr hohe spezifische Varianz (geringe Kommunalität) aufweist
# können wir auch ein 3-Faktor-Modell ohne diesem Merkmal anschauen.
# Dieses kann den RMSE um knapp 44% reduzieren (im Vergleich sind die iterierten PAF-Methoden),
# weshalb wir für dieses Merkmal für die weitere Analyse entfernen.

# In[20]:


X_without_age = X.drop(columns="housing_median_age", axis=1)
fa_without_age = FactorAnalysis(n_factors=3).fit(X_without_age)
perc = 1 - fa_without_age.get_rmse() / rmse
print(f"Der RMSE konnte durch Entfernen des Merkmals um {perc:.2%} reduziert werden")


#
# Den Unterschied zwischen den Methoden im RMSE können wir auch noch grafisch analysieren:

# In[21]:


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


# In[22]:


# Zusammenfassung der iterierten PAF-Methode
methods[2][1].print_summary()


# In der Zusammenfassung können wir neben den Ladungen auch die Kommunalitäten und die spezifischen Varianzen
# der Merkmale sehen.
#
#
# Die spezifische Varianz von `housing_median_age` ist mit einem Wert von 0.90 sehr hoch.
# Das bedeutet, dass die Faktoren die Varianz dieses Merkmals gemeinsam nicht besonders gut erklären können. Dies spiegelt sich ebenfalls in den
#  geringen Ladungen auf die drei Faktoren wider.
# Die spezifischen Varianzen bei den restlichen Merkmalen sind jedoch sehr niedrig,
# was ein gutes Zeichen für die Qualität der Faktorlösung ist.
#
# Alle Merkmale können eindeutig einem Faktor durch jeweils die betraglich
# größte Ladung zugeordnet werden. Dies ist nicht immer der Fall, sodass eine Faktorrotation für eine leichtere Interpretation
# sorgen könnte. Hier ist jedoch auch ohne Rotation eine Interpretation gut möglich. Wir werden im nächsten Schritt trotzdem
# beispielhaft die mit der Varimax-Methode rotierten Faktorladungen ansehen.
#
# Bevor wir das tun, werden wir noch die verschiedenen initialen Schätzungen der Kommunalitäten in der (iterierten) PAF-Methode vergleichen.
# Interessant könnte dabei sein, ob die Wahl der initialen Schätzung einen Einfluss auf die finalen Ladungen hat.

# In[23]:


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
# Unterschiede, vor allem im zweiten Faktor feststellen. Beispielsweise hat der zweite Faktor hier ein unterschiedliches
# Vorzeichen, jedoch nur wenn Einsen als Kommunalitätsschätzung benutzt wurden. In diesem
# Fall ist das Ergebnis identisch zur Hauptkomponentenmethode.
#
# Wir stellen fest, dass die iterierte Variante jedoch eine unterschiedliche Anzahl an Iterationen benötigt, bis
# das Konvergenzkriterium erreicht wird. Am langsamsten ist es auf diesem Datensatz im Falle von Einsen (10 Iterationen) und am schnellsten ist es
# bei den maximalen absoluten Korrelationen (MAC) (nur 3 Iterationen). Ist das Konvergenzkriterium erfüllt, sind die Ladungen jedoch
# bei allen drei initialen Schätzungen weitestgehend identisch.

# # Faktoranalyse Schritt 4: Faktorrotation und -interpretation
# Jetzt rotieren wir die Ladungen mit der Varimax-Methode und versuchen, die Faktoren zu interpretieren.

# In[24]:


methods = [
    ("Unrotiert", FactorAnalysis(method="paf", rotation=None)),
    ("Varimax-Rotation", FactorAnalysis(method="paf", rotation="varimax")),
]
fa_params = {"n_factors": 3}
fig = create_loadings_heatmaps(X_without_age, methods, fa_params=fa_params)
plt.tight_layout()


# Müssten wir die Faktoren interpretieren, könnte man sagen, dass
# - der erste Faktor die Größe des Bezirks widerspiegelt,
# - der zweite Faktor den Standort des Bezirks berücksichtigt und
# - der dritte Faktor den Wohlstand im Bezirk bezeichnet.
#

# # Faktoranalyse Schritt 5: Die Faktorwerte bestimmen
#
# Als letzten Schritt können wir noch einen Blick auf die geschätzten Faktorwerte
# werfen. Wie hätte als der Bezirk $i$ die drei oben beschriebenen Faktoren bewertet?
#
# Die hier benutzte Methode zur Schätzung der Faktorwerte ist die *Regressionsmethode*,
# welche eine multivariate lineare Regression benutzt, um die Faktorwerte
# zu schätzen.

# In[25]:


scores = FactorAnalysis(n_factors=3).fit_transform(X_without_age)
scores = pd.DataFrame(scores, columns=["Größe", "Standort", "Wohlstand"])
scores.head()


# In[26]:


scores.std(axis=0)


# Hier weisen die Faktorwerte keine Einheitsvarianz beziehungsweise Standardabweichung von eins auf, weil die PAF-Methode mit
# quadrierten multiplen Quadraten als initiale Schätzung verwendet wurde.
#
# Benutzt man jedoch die Hauptkomponentenmethode, so weisen die Faktorwerte eine
# Standardabweichung von Eins auf.

# In[27]:


scores = FactorAnalysis(method="pc", n_factors=3).fit_transform(X)
scores.std(axis=0)
