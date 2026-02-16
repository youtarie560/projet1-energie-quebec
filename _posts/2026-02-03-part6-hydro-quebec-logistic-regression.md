---
layout: post
title: "Partie 6: Modèle combiné"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---
Nous avons assemblé toutes les pièces du puzzle :
1.  **Données :** Nettoyage et séparation temporelle.
2.  **Physique :** Ajout des degrés-jours (HDD) et inertie (Lags).
3.  **Régularisation :** Ridge ($\lambda=100$) pour gérer la multicolinéarité.
4.  **Stacking :** Ajout de la probabilité de pointe ($P_{pointe}$) comme signal d'alerte.

Contrairement à notre hypothèse initiale, l'ajout de la couche de classification (Stacking) n'a pas amélioré la performance 
globale sur le jeu de test.

### 6.1 Performance Finale
Le modèle combiné atteint sa meilleure performance :

| Étape                              | $R^2$ (Test) | RMSE (Test) | Gain                      |
|:-----------------------------------|:-------------|:------------|:--------------------------|
| 1. OLS Baseline                | 0.6723       | 40.5        | -                         
| 2. Ridge + Features (Partie 4) | 0.6777       | 40.17   |  +0.80% (Meilleur Modèle) |
| 3. Ridge + Stacking (Partie 5)     | 0.6735       | 40.43       | -0.62% (vs Ridge)         


### Analyse des Résidus
L'ajout de $P(pointe)$ a légèrement dégradé le modèle (-0.41% de $R^2$). Cela s'explique par deux facteurs observés dans nos données:
1. La moyenne de $P(pointe)$ sur le jeu de test est extrêmement faible (0.003). Il n'y avait quasiment aucune "crise" à détecter sur cette période.
2. Puisqu'il n'y avait pas de signal fort (pas de vraie pointe), la variable probabiliste $P(pointe)$ a agi comme du bruit supplémentaire, 
que le modèle Ridge a eu du mal à filtrer complètement, augmentant légèrement l'erreur moyenne (RMSE passe de 40.17 à 40.43).

Conclusion intermédiaire : Le modèle le plus robuste est le Ridge Standard (Partie 4). La complexité additionnelle du Stacking n'est 
justifiée que si la période de test contient des événements extrêmes, ce qui n'était pas le cas ici.

### 6.2. Analyse des Résidus
Histogramme (Gauche)
* La distribution des erreurs suit une courbe en cloche (Gaussienne) centrée sur 0
  * le modèle ne fait pas d'erreur systématique (biais nul). Il ne surestime ni ne sous-estime la consommation en moyenne.

Prédictions vs Réel (Droite)
* Le nuage de points est dense autour de la diagonale ($y=x$).
* On observe une dispersion due à la variabilité humaine (RMSE $\approx$ 40 kWh), mais pas de structure non-linéaire que le modèle aurait ratée.

![Analyse des résidus](assets/images/model_final_analysis.png)

Pour un déploiement réel, nous recommandons le modèle Ridge (Partie 4). Il est plus simple, plus interprétable, et offre
la meilleure performance (RMSE la plus basse) sur les données de test.


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
