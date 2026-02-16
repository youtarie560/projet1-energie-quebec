---
layout: post
title: "Partie 5: Sous-tâche de classification"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---
Notre modèle Ridge précédent (Partie 4) est un "généraliste" qui cherche à minimiser l'erreur moyenne. Par nature, il lisse 
les extrêmes et sous-estime souvent les comportements anormales comme les pointes de consommation critiques.

Pour remedier ceci, nous avons mis en place une sous-tâche de classification (Stacking). Ceci consiste à:
1. Entraîner un "spécialiste" (Régression Logistique) pour détecter uniquement les événements de pointe 
2. Nous avons utiliser le paramètre `class_weight='balanced'` pour forcer ce modèle à être sensible aux anomalies
(les pointes étant rares). 
3. Injecter la probabilité prédite ($P(\text{Pointe})$) comme une nouvelle variable explicative dans le modèle de régression final

En théorie, c'est comme si le modèle de régression consultait un spécialiste avant de donner sa réponse finale.
* Si $P(\text{Pointe})$ est faible, Ridge suit la température et l'historique.
* Si $P(\text{Pointe})$ est élevé, Ridge reçoit un signal fort (via un coefficient positif) pour augmenter sa prévision, simulant la surcharge du réseau.

### 5.1. Analyse du Classifieur
* Accuracy Train (0.84) vs Test (0.94)
  * Le modèle généralise bien, ce qui est rassurant.
  * Moyenne $P(\text{Pointe})$ (0.18) : Alors que les vraies pointes représentent moins de 1% des données, notre classifieur
signale un risque significatif dans 18% des cas. 
    * Il ne cherche pas à avoir raison tout le temps, il cherche à ne jamais rater une crise, quitte à donner de fausses alertes

Cette technique nous a permis d'améliorer le RMSE final et de mieux capturer les pics hivernaux.

### Résultats du Classifieur
L'ajout de la probabilité de pointe `P_pointe` a un impact sur le modèle final, mais son coefficient est **-1.2437**
Cela semble contre-intuitif (pourquoi un risque de pointe baisserait-il la prévision ?), mais cela s'explique par la 
colinéarité avec les autres variables. 
* Les variables comme `energie_lag24`, `degre_jour_chauffage` ou `clients_connectes` poussent déjà la prédiction très haut quand il
fait froid.
* Il est probable que ces variables surestiment légèrement la consommation dans certaines situations de haute tension
* Le modèle utilise donc $P(\text{Pointe})$ comme un facteur de correction pour affiner la prédiction vers le bas (-1.24 unités)
afin d'être plus précis. 


#### 5.3. Comparaison de Performance Finale

| Modèle | $R^2$ (Test) | RMSE (Test) 
| :--- |:-------------|:------------|
| Ridge (Partie 4) | 0.6777       | 40.17
| **Ridge + Stacking (Partie 5)**| 0.6759       | 40.28 

### Conclusion
Le gain de performance est minime, voire légèrement négatif sur le RMSE global (de 40.17 à 40.28).

### Question
1. Pourquoi utiliser P(pointe) au lieu d’un indicateur 0/1?

Transformer une probabilité continue en une décision binaire (0 ou 1) détruit l'information sur la certitude du modèle.
* Cas A
  * ($P=0.51$) : Le classifieur est incertain. C'est "peut-être" une pointe
* Cas B 
  * ($P=0.99$) : Le classifieur est formel. Avec un seuil binaire à 0.5, ces deux situations deviendraient identiques (1). 

En utilisant la probabilité brute, nous permettons au modèle de régression d'ajuster sa réponse proportionnellement au risque.
Avec notre coefficient de -1.24, voici comment le modèle réagit:
* Risque Faible ($P=0.1$) : Correction = $-1.24 \times 0.1 = \mathbf{-0.12}$. (Correction négligeable)
* Risque Moyen ($P=0.5$) : Correction = $-1.24 \times 0.5 = \mathbf{-0.62}$. (Ajustement modéré)
* Risque Critique ($P=0.9$) : Correction = $-1.24 \times 0.9 = \mathbf{-1.11}$. (Correction maximale)


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