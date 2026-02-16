---
layout: post
title: "Partie 7: Extension"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---

### Option C (Analyse d’erreur approfondie)
### 6.1 Dépendance à la Température
Notre tentative d'amélioration via le Stacking (Partie 6) n'ayant pas amélioré le score $R^2$ (-0.41%), il devient inutile 
d'ajouter aveuglément de la complexité ou des données externes sans comprendre la source du problème. 
Nous avons donc choisi d'analyser la structure des erreurs (résidus) pour identifier les faiblesses du modèle linéaire.

Nous avons calculé la corrélation entre l'Erreur Absolue ($|y_{pred} - y_{true}|$) et la Température Extérieure
* Corrélation de -0.3334
* Cela signifie que plus la température baisse, plus l'erreur augmente
* Le modèle souffre d'hétéroscédasticité. Il prédit très bien par temps doux (printemps/automne), mais perd en précision lors des 
grands froids hivernaux
  * Cause probable : La relation entre le froid extrême et la consommation n'est pas parfaitement linéaire (même avec nos HDD).
À très basse température, les systèmes de chauffage auxiliaires (plinthes électriques) s'activent de manière agressive, et l'efficacité des thermopompes chute, créant une demande exponentielle que le modèle Ridge (linéaire) sous-estime souvent.

### 6.2 Dépendance Temporelle (Le problème de Minuit)

Nous avons agrégé l'erreur moyenne par heure de la journée pour voir si le modèle échoue à certains moments clés.

* L'heure la plus difficile à prédire est 0h00 (Minuit). 
* Analyse 
  * Minuit représente une "heure charnière". C'est souvent le moment où les thermostats programmables basculent en mode "Nuit" (baisse de la consigne) et où, inversement, certains appareils automatisés (chauffe-eau, recharge de véhicules électriques) peuvent se déclencher. 
  * Le modèle se base beaucoup sur l'inertie (Lag 24h), a du mal à anticiper exactement quand cette transition de consommation se produit d'un jour à l'autre.

Cette analyse révèle que notre modèle n'est pas "mauvais partout", mais qu'il est spécifiquement faible lors des nuits très froides.
Pour une version future, nous ne recommanderions pas d'ajouter plus de données météo, mais plutôt de changer l'architecture du modèle.
* Utiliser un algorithme basé sur les arbres (ex: XGBoost ou Random Forest, ou autre modèle non-linéaire) qui peut apprendre les seuils de rupture (ex: "Si T < -20°C, alors la consommation double")
* Entraîner un modèle spécifique pour la nuit (23h-5h) et un autre pour le jour





















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
