---
layout: post
title: "Partie 4: Régression Ridge"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---

Avec l'ingénierie des caractéristiques de la Partie 3, nous avons donné une "mémoire" et une "intelligence physique"
à notre modèle. Cependant, nous avons aussi créé un problème mathématique : la **multicolinéarité**.
1. ```temperature_ext``` et ```degre_jour_chauffage``` sont fortement inversement corrélées (quand l'une monte, l'autre descend)
2. ```energie_lag1``` et ```energie_rolling_mean_6h``` racontent presque la même histoire. C'est-à-dire que la consommation d'il y a 1h est proche de la moyenne des dernières 6h


OLS gère mal ce dernier car il a tendance à attibuer des coefficients énormes et opposés pour compenser
(imaginez deux personnes qui tirent sur une corde dans des directions opposées), ce qui rend le modèle instable.
Nous appliquerons une technique appelée Ridge **(Régularisation L2)**.

### Pourquoi Ridge ?
La régression Ridge ajoute une pénalité ($L_2$) à la fonction de coût :
$$J(\theta) = \text{MSE} + \lambda \sum \theta_i^2$$

Cela force le modèle à "rétrécir" les coefficients inutiles vers zéro et à répartir le poids entre les variables corrélées.
### Standardisation
Avant d'apliquer Ridge, nous avons utilisé ```StandardScaler``` pour transfomer toutes les variables pour qu'elles aient une moyenne de 
0 et un écart-type de 1 ($$ z = \frac{x - \mu}{\sigma} $$). Par exemple, si ```clients_connectes``` ~ 100 et ```heure_sin``` ~ 0.7, 
Ridge va écraser ```heure_sin``` injustement.
```python
# scaling avant
scaler = StandardScaler()
X_train_reg = scaler.fit_transform(X_train_raw)
X_test_reg = scaler.transform(X_test_raw)
```
La régression Ridge cherche à minimiser la somme des coefficients au carré $$\lambda \sum \theta^2$$.

### Validation Croisée Temporelle (TimeSeriesSplit)
Pour trouver le bon paramètre de pénalité $\lambda$ idéal, nous ne pouvons pas utiliser une validation croisée aléatoire (k-fold), 
car cela briserait la chronologie. Nous utilisons **TimeSeriesSplit** où:
* entraîne sur [Jan-Mars] 
* valide sur [Avril], puis on entraîne sur [Jan-Avril] et on valide sur [Mai], etc.

### Résultats

On observe que le modèle performe remarquablement bien sur le jeu de test. Le fait que la métrique $R^2$de test soit supérieur 
à la métrique d'entraînement suggère que la période de test est plus "stable" ou cyclique que la période d'entraînement.

| Modèle                  | $R^2$ (Train data) | $R^2$ (Test data) | RMSE (Test data) | Interprétation                            |
|:------------------------|:-------------------|:------------------|:-----------------|:------------------------------------------|
| **OLS (Baseline)**      | 0.5198             | 0.6723            | 40.5             | Bonne performance de base                 |
| **Ridge $\lambda=100$** | 0.5194             | **0.6777**        | **40.17**        | légère amélioration et meilleur stabilité |

*Tableau 1 : Comparaison de performance (OLS vs Ridge)* <br>
Bien que le gain en $R^2$ semble modeste (+0.005), Ridge a réussi à nettoyer le modèle en réduisant les coefficients des variables redondantes.
#### Ridge "punit" la redondance
C'est ici que Ridge montre toute sa puissance. Il a détecté que plusieurs variables racontaient la même histoire et a 
"éteint" les moins pertinentes pour se concentrer sur les plus robustes.

| Caractéristique | Coefficient OLS       | Coefficient Ridge | Réduction (%) | Analyse |
| :--- |:----------------------|:------|:--------------|:--------| 
| `indice_temp_cons` | -9.47                 | -1.16 | -87.8%        |   Ridge a compris que cette info est déjà dans ```indice_temp_client```    |
| `temperature_ext` | 62.12                 | 9.79  | -84.2%        | Ridge préfère les variables physiques (HDD/CDD) à la température brute.        |
| `degre_jour_chauffage` | 73.72                 | 15.04 | -79.6%        |Réduit au profit de l'interaction avec les clients.         |
| `est_ferie` | -0.05                 | 0.03  | -50.4%        |     Impact négligeable    |

*Tableau 2: Top 5 des réductions de coefficients* <br>
En regardant les coefficients, vous remarquerez que certaines valeurs sont négatives. Cela ne signifie pas que ces
caractéristiques sont inutiles. Au contraire, cela indique une **relation inverse**.

Interprétation : OLS donnait un poids énorme à la température (62.12) et au chauffage (73.72). Ridge a calmé le jeu en réduisant la température à 9.79, laissant les variables d'interaction plus complexes prendre le relais.

Maintenant il faut se demander: Quelles sont les variables qui survivent à la régularisation et pilotent la prédiction?

| Rangs                  | Caractéristique       | Coefficient Ridge | Rôle Physique
|:-----------------------|:----------|:------|:--------------|:--------
| 1     | `energie_lag24`    | 79.05 | -87.8%        |Mémoire (Inertie)
| 2      | `indice_temp_client`   | 38.00  | -84.2%        |Physique (Météo $\times$ Taille Réseau)
| 3| `heure_cos`        | -31.19 | -79.6%        | Cycle Journalier (Jour/Nuit)
| 4         | `clients_connectes`   | 29.18  | -50.4%        | Échelle de la demande
| 5       | `heure_sin`   | -23.04  | -50.4%        |  Cycle Journalier


Avec un coefficient de 79.05, la consommation de la veille (`energie_lag24`) est de loin le meilleur prédicteur. 
Cela confirme que l'inertie du réseau est plus forte que la météo
#### 2. Les vrais moteurs de la consommation (MVP)
Quelles sont les variables les plus importantes pour le modèle final ? Ce n'est pas la météo, mais l'inertie du système.

*Top 5 des coefficients les plus importants:*

| Caractéristique                | Coefficient Ridge | Rôle                 |
|:-------------------------------|:------------------|:---------------------|
| 1. `degre_jour_chauffage`      | 71.57             | Physique (Froid)     |
| 2. `energie_rolling_mean_6h` | 64.21             | Tendance (Inertie) |
| 3. `clients_connectes`         | 35                | Taille du réseau     |
| 4. `heure_cos`                 | -34.52            | Cycle journalier     |
| 5. `energie_lag1`              | 32.5805        | Mémoire immédiate    |


## Conclusion
L'utilisation de la régression Ridge a permis de transformer un modèle potentiellement instable (à cause des nombreuses variables corrélées créées en Partie 3) 
en un modèle robuste.

1. Ridge a réduit les coefficients conflictuels (Température vs Chauffage) de près de 80% 
2. Ridge a révélé que `energie_lag1` et la taille du réseau sont les facteurs dominants
3. Avec un RMSE de **40.17**, le modèle est précis et généralise bien sur les données de test.

## Questions
### 1. Pourquoi Ridge aide-t-il quand les caractéristiques sont corrélées?
Prenons l'exemple de `temperature_ext` ($T$) et `degre_jour_chauffage` ($HDD$) qui sont fortement inversement corrélées (quand $T$ monte, $HDD$ descend).
Imaginez que la relation physique soit que la consommation baisse de -10 kWh pour chaque degré de température supplémentaire.
* Sans Ridge sur OLS : Le modèle peut devenir "fou" pour atteindre ce résultat. Il pourrait dire:
  * Coefficient $T$ : +1000
  * Coefficient $HDD$ : +1010
  * Si $T$ augmente de +1, alors $HDD$ diminue d'environ -1. L'effet total est $1000(1) + 1010(-1) = -10$
    * Le résultat est correct, mais les coefficients (+1000 et +1010) sont énormes et instables. Au moindre bruit dans les données, ils pourraient passer à +5000 et +5010.
* Avec Ridge sur OLS: La pénalité $\lambda$ interdit ces coefficients géants. Ridge va forcer le modèle à choisir des valeurs plus 
petites, par exemple
  * Coefficient $T$ : -4
  * Coefficient $HDD$ : +6
    * $-4(1) + 6(-1) = -10$. 
      * Le résultat est le même, mais les coefficients sont petits, stables et physiquement interprétables.

Bref, il peut donner +1000 à l'une et -995 à l'autre pour compenser mais ceci crée une variance énorme: si on change un tout petit 
peu les données d'entraînement, les coefficients explosent. Ridge impose une contrainte sur des coefficients élevés.
      
Voici une explication Mathématique: 
La solution des Moindres Carrés Ordinaires (OLS) est donnée par $$\hat{\beta}_{OLS} = (X^T X)^{-1} X^T y$$
 Le problème survient lors de l'inversion de la matrice $(X^T X)$. Si des colonnes de $X$ sont colinéaires (corrélées), 
 la matrice $X^T X$ devient singulière (ou mal conditionnée). Cela signifie que son déterminant est proche de zéro. 
 Par conséquent, son inverse $(X^T X)^{-1}$ ne peut pas être calculée correctement (ou devient extrêmement instable avec des valeurs numériques gigantesques), 
 ce qui fait "exploser" la variance des coefficients $\beta$. Ridge ajoute un terme de régularisation $\lambda I$ (la matrice identité) sur la diagonale: 
    $$\hat{\beta}_{Ridge} = (X^T X + \lambda I)^{-1} X^T y$$ L'ajout de $\lambda$ (un nombre positif) sur la diagonale de la matrice garantit qu'elle devient inversible (non-singulière), stabilisant ainsi mathématiquement le calcul de $\beta$.

### 2. Quelle caractéristique a été la plus réduite? Pourquoi?
Selon nos résultats:<br> 
 Les meilleures réduction sont `indice_temp_cons` (-87.8%) et `temperature_ext` (-84.2%). `degre_jour_chauffage` a aussi 
été massivement réduit (-79.6%). 
* cas `indice_temp_cons`:
  * Nous avons donné au modèle deux variables très proches:
    * `indice_temp_cons` 
    * `indice_temp_client` (L'indice thermique $\times$ le nombre de clients).
  * Ridge a compris que l'impact de la météo dépend de la taille du réseau. La variable `indice_temp_client` 
(qui a gardé un coefficient fort de 38.0) contient plus d'information que l'indice seul.
  * Par conséquent, Ridge a considéré `indice_temp_cons` comme un doublon inutile et a écrasé son coefficient vers 0 
pour laisser la place à la variable d'interaction plus précise
* cas `temperature_ext` :
  * Dans la Partie 1 (OLS simple), la température brute était la seule information disponible, donc son coefficient était énorme.
  * Dans la Partie 3, nous avons introduit des transformations physiques : $HDD$ et les indices thermiques.
  * Mathématiquement, la corrélation est extrême : $HDD \approx -1 \times Temperature$.
  * Le HDD (qui vaut 0 quand il fait chaud) est une bien meilleure représentation de la réalité physique du chauffage que 
la température brute (qui continue de varier inutilement en été). Ridge a donc transféré le poids de la prédiction de la variable
"bruitée" (temperature_ext) vers les variables "physiques" (indice_temp_client et degre_jour_chauffage), réduisant
l'importance de la température brute.

### 3. Comment interpréter Ridge comme estimation MAP ?
La régression Ridge est mathématiquement équivalente à une estimation du Maximum A Posteriori (MAP) dans un cadre Bayésien.
L'approche Bayésienne combine les données observées avec une croyance préalable.
1. $P(y|X, \beta)$ On suppose que le lien entre $X$ et $y$ est linéaire avec un bruit Gaussien (Loi Normale): $y = X\beta + \epsilon$, 
où $\epsilon \sim \mathcal{N}(0, \sigma^2)$.La probabilité d'observer nos données $y$ sachant un modèle $\beta$ est:
$$P(y|X, \beta) \propto \exp\left( -\frac{||y - X\beta||^2}{2\sigma^2} \right)$$
Maximiser ce terme revient à faire OLS.
2. A Priori est $P(\beta)$
C'est ici qu'intervient Ridge. Avant même de voir les données, nous avons une croyance: nous pensons que les coefficients $\beta$
doivent être petits pour éviter le surapprentissage. On modélise cette croyance par une loi Normale centrée sur zéro avec 
une variance $\tau^2$ :$$P(\beta) \sim \mathcal{N}(0, \tau^2) \propto \exp\left( -\frac{||\beta||^2}{2\tau^2} \right)$$
3. A Posteriori est $P(\beta|y, X)$
Selon la règle de Bayes, la probabilité finale de notre modèle est proportionnelle au produit de la Vraisemblance et du Priori
$$\underbrace{P(\beta|y, X)}_{\text{Posteriori}} \propto \underbrace{P(y|X, \beta)}_{\text{Vraisemblance}} \times \underbrace{P(\beta)}_{\text{Prior}}$$
4. Pour trouver le meilleur $\beta$ (Estimation MAP), on cherche à maximiser cette probabilité a posteriori. 
Pour simplifier le calcul des exponentielles, on prend le logarithme négatif (ce qui transforme la maximisation en minimisation) <br>
$$\text{MAP} = \arg\min_{\beta} \left( -\log(P(y|X, \beta)) - \log(P(\beta)) \right)$$ <br>
En remplaçant par les formules gaussiennes, on obtient exactement la fonction de coût Ridge
$$\hat{\beta}_{MAP} = \arg\min_{\beta} \left( \frac{1}{2\sigma^2} ||y - X\beta||^2 + \frac{1}{2\tau^2} ||\beta||^2 \right)$$
<br> Cela revient à minimiser $$||y - X\beta||^2 + \lambda ||\beta||^2$$ <br>(Où $\lambda = \frac{\sigma^2}{\tau^2}$, le ratio entre le bruit 
des données et notre incertitude a priori).

En résumé: Utiliser Ridge, c'est adopter une posture Bayésienne qui dit : "Je crois a priori que mes coefficients doivent être petits 
et centrés autour de 0, à moins que les données ne prouvent le contraire avec force (Vraisemblance)."











<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']],
      displayMath: [['$$', '$$'], ['\\[', '\\]']],
      processEscapes: true
    }
  };
</script>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>