---
layout: post
title: "Partie 3: Ingénierie des Caractéristiques (Feature Engineering)"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---

Pour capturer la complexité de la demande énergétique et améliorer la performance de nos modèles linéaires, nous avons enrichi
le jeu de données initial en créant de nouvelles variables explicatives. Ces transformations visent à modéliser la mémoire 
du système et les lois physiques non-linéaires du chauffage.

### 3.1. La Mémoire du Système (Lags)
La consommation électrique présente une forte autocorrélation temporelle. Pour exploiter cette structure, nous avons 
introduit des variables retardées ("lags") en prenant soin de séparer les calculs par station ```poste A``` , ```poste B```, ```poste C```
pour éviter toute fuite de données entre sites. Les modèles mathématiques (comme la régression linéaire ou Ridge) ne comprennent pas le texte. 

* Lag 1h (```energie_lag1```): La consommation de l'heure précédente. C'est l'indicateur le plus direct de l'état actuel du réseau $$X_{t-1}$$
* Lag 24h Pondéré (```energie_lag24```): Au lieu d'utiliser une simple valeur retardée de 24h (souvent bruitée), nous avons conçu une fenêtre glissante de 
48h centrée sur l'heure de la veille, pondérée par une fonction exponentielle. Cela permet de capturer la tendance journalière
 tout en donnant plus d'importance aux heures les plus proches de l'heure cible la veille.
  * Utilisation de scipy.signal.windows.exponential sur une fenêtre temporelle de 48h ```.rolling(window="48h")```

### 3.2 Modélisation Physique de la Température
La relation entre la température extérieure et la consommation électrique n'est pas linéaire. 
Un modèle linéaire classique ($y = ax + b$) échoue à capturer le point de bascule où le chauffage s'allume.
* Degrés-Jours de Chauffage (HDD - ```degre_jour_chauffage```) :
  * Nous avons appliqué une transformation "rectifier" (ReLU) avec un seuil de 18°C.$$HDD = \max(18 - T_{ext}, 0)$$
    * Si $T_{ext} > 18^\circ C$ : HDD = 0 (Pas de chauffage).
    * Si $T_{ext} < 18^\circ C$: La demande de chauffage augmente linéairement avec le froid. Cette variable permet au modèle 
linéaire de "s'éteindre" en été et de "s'activer" proportionnellement en hiver.
* Degrés-Jours de Climatisation (CDD - ```degre_jour_clim```):
  * Similaire au HDD, mais pour la climatisation avec un seuil de 23°C
    * $$CDD = \max(T_{ext} - 23, 0)$$

### 3.3 Modélisation Physique de la Température
Pour affiner la modélisation, nous avons créé des variables d'interaction qui capturent des phénomènes physiques.
* Indice de Confort Thermique Pondéré (```indice_temp_cons```)
  * Le chauffage consomme généralement plus d'énergie que la climatisation pour un même écart de température. Nous avons créé un indice quadratique pondéré
    * $$\text{Indice} = (2 \times HDD + 0.65 \times CDD)^2$$
  * Le terme quadratique reflète que la consommation accélère lors des froids extrêmes (résistance des matériaux, efficacité des pompes à chaleur qui chute).

* Facteur Éolien (```facteur_eolien```)
  * Le vent affecte la thermique des bâtiments (effet de refroidissement éolien). Nous avons modélisé cette interaction 
en multipliant notre indice thermique par la vitesse du vent.
    * $$\text{Facteur Éolien} = \text{Indice Thermique} \times V_{vent}$$
  * Cela aide le modèle à distinguer un froid calme (-10°C, vent nul) d'un froid venteux (-10°C, 50 km/h) qui demande beaucoup plus d'énergie.

### 3.4 Interaction Taille Réseau
L'impact de la météo sur la consommation totale est proportionnel à la taille du réseau local. Une vague de froid génère une 
demande énergétique beaucoup plus importante sur un secteur comportant 1000 clients que sur un secteur n'en comportant que 50. 
Donc, Nous avons multiplié l'indice thermique par le nombre de clients connectés pour ajuster l'échelle de la réaction.
* Interaction Taille Réseau (```indice_temp_client```)
  * $$\text{Indice}_{client} = (2 \cdot HDD + 0.65 \cdot CDD)^2 \times \text{Clients Connectés}$$
  * Cette variable permet au modèle de différencier la réaction thermique entre les secteurs résidentiels denses (Poste A ou C) et les secteurs plus petits ou industriels (Poste B).

### Conclusion
Nous avons sélectionné un ensemble précis de nouvelles variables pour enrichir le jeu de données original. 
Certaines pistes explorées (comme ```cycle_clients``` ou l'encodage manuel des postes) ont été écartées pour éviter la 
redondance ou la multicolinéarité.
Voici la liste finale des variables techniques (features_eng) intégrées au modèle :
```python
features_eng = [
    # --- MÉMOIRE (LAGS) ---
    'energie_lag1',          # Consommation de l'heure précédente (Inertie immédiate)
    'energie_lag24',         # Consommation de la veille (Habitudes journalières, lissé)

    # --- PHYSIQUE (NON-LINÉARITÉ) ---
    'degre_jour_chauffage',  # Besoin de chauffage (si T < 18°C)
    'degre_jour_clim',       # Besoin de climatisation (si T > 23°C)

    # --- INTERACTIONS COMPLEXES ---
    'indice_temp_cons',      # Indice quadratique pondéré (Chauffage > Clim)
    'indice_temp_client',    # Impact thermique ajusté à la taille du réseau
    'facteur_eolien'         # Effet refroidissant du vent sur le bâtiment
]
```



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