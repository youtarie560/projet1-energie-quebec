---
layout: post
title: "Partie 2: Régression logistique avec descente de gradient"
date: 2026-02-03 12:00:00 -0500
categories: machine-learning project
use_math: true
---
Cette section se concentre sur la régression logistique. Contrairement à la régression linéaire qui prédit une valeur continue,
la régression logistique est utilisée pour la classification binaire (0 ou 1).
Le cœur du modèle repose sur la fonction sigmoïde: $$\sigma(z) = \frac{1}{1 + e^{-z}}$$.
La sigmoïde transforme un score réel $z = \theta^T x$ en une probabilité dans l'intervalle $[0, 1]$. 
Comme vu en classe, elle approxime la fonction échelon (step function) tout en restant différentiable, 
ce qui est essentiel pour la descente de gradient.

### La Fonction de Perte (Log Loss)
Pour entraîner ce modèle, nous ne pouvons pas utiliser les moindres carrés (MSE) car la fonction n'est pas convexe pour la classification. 
Nous utilisons plutôt l'entropie croisée binaire (Binary Cross-Entropy), qui découle du principe du Maximum de Vraisemblance:
$$J(\theta) = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$ <br> 

### Implémentation Python
Voici mon implémentation "from scratch". J'ai utilisé ```np.clip``` pour éviter les erreurs numériques (comme $$log(0)$$).
Le gradient utilisé pour la mise à jour des poids est donné par la formule suivante:<br>
$$\nabla J(\theta) = \frac{1}{N} X^T (\hat{y} - y)$$

```python
def sigmoid(z):
    """
    Fonction sigmoïde.

    Indice: Pour la stabilité numérique, clip z entre -500 et 500.
    """
    z_clipped = np.clip(z, -500, 500)
    return 1 / (1 + np.exp(-z_clipped))

def cross_entropy_loss(y_true, y_pred_proba):
    """
    Calcule la perte d'entropie croisée binaire.

    Indice: Clip les probabilités pour éviter log(0).
    """
    epsilon = 1e-15
    clipped_log = np.clip(y_pred_proba, epsilon, 1-epsilon)

    return -np.mean(y_true*np.log(clipped_log) + (1-y_true)*np.log(1-clipped_log))


def logistic_gradient(X, y, beta):
    """
    Calcule le gradient de la perte d'entropie croisée.

    Paramètres:
        X : ndarray (n, p+1) - caractéristiques AVEC colonne de 1
        y : ndarray (n,) - étiquettes binaires
        beta : ndarray (p+1,) - coefficients actuels

    Retourne:
        gradient : ndarray (p+1,)
    """

    z = X @ beta
    erreur = sigmoid(z) - y

    return (1/len(y)) * (X.T @ erreur)

def logistic_fit_gd(X, y, lr=0.1, n_iter=1000, verbose=False):
    """
    Entraîne la régression logistique par descente de gradient.

    Paramètres:
        X : ndarray (n, p) - caractéristiques SANS colonne de 1
        y : ndarray (n,) - étiquettes binaires (0 ou 1)
        lr : float - taux d'apprentissage
        n_iter : int - nombre d'itérations
        verbose : bool - afficher la progression

    Retourne:
        beta : ndarray (p+1,) - coefficients [intercept, coef1, ...]
        losses : list - historique des pertes
    """
    # 1. Ajouter colonne de 1 à X
    n, p = X.shape
    ones_col = np.ones((n, 1))
    X_aug = np.hstack([ones_col, X])

    # 2. Initialiser beta à zéro
    # Taille = nombre de features + 1 (pour l'intercept)
    beta = np.zeros(p + 1)

    losses = []
    # 3. Boucle de descente de gradient
    for i in range(n_iter):
        grad = logistic_gradient(X_aug, y, beta)

        beta = beta - (lr * grad)
        if i % 100 == 0 or i == n_iter - 1:
            probas = sigmoid(X_aug @ beta)
            loss = cross_entropy_loss(y, probas)
            losses.append(loss)
            if verbose:
                print(f"Iteration {i}: Loss = {loss:.4f}")

    # 4. Retourner beta et historique des pertes
    return beta, losses


def logistic_predict_proba(X, beta):
    """
    Prédit la probabilité P(Y=1 | X).
    """
    # Il faut ré-ajouter la colonne de 1 car X arrive sans intercept
    n, _ = X.shape
    ones_col = np.ones((n, 1))
    X_aug = np.hstack([ones_col, X])

    return sigmoid(X_aug @ beta)
```


## Convergence de l'entraînement
Nous avons entraîné le modèle en utilisant la **Descente de Gradient** sur 500 itérations. Comme le montre la figure ci-dessous, la fonction de perte (entropie croisée) diminue régulièrement, ce qui confirme que notre implémentation manuelle du gradient est correcte et que le modèle apprend.
![Convergence Plot](assets/images/Code_Generated_Image.png)
*Figure 1 : Diminution de la perte d'entropie croisée au fil des itérations.*

## Évaluation et le Paradoxe de l'Exactitude
Avant d'analyser les résultats, il est important de comprendre ce que nous essayons de prédire. 
Dans le réseau d'Hydro-Québec, une **Pointe de consommation** survient lors des grands froids hivernaux (ex: -20°C ou -30°C), 
lorsque tous les foyers utilisent le chauffage électrique simultanément.

Ces moments sont critiques pour 2 raisons :
1.  Si la demande dépasse l'offre, cela peut causer des pannes généralisées.
2.  Pour répondre à cette demande soudaine, Hydro-Québec doit parfois importer de l'énergie coûteuse ou utiliser des centrales complémentaire.

Notre modèle de classification a donc une mission de "détection d'anomalie" : 
il doit repérer ces rares moments de stress intense du réseau (qui apparaissent seulement 0.9% des données) parmi des milliers d'heures de consommation normale.
Après l'entraînement, nous avons évalué le modèle sur l'ensemble de test.

| Métrique                              | Valeur |
|---------------------------------------|-------|
| Exactitude (Accuracy - Training data) | 99.05% |
| Exactitude (Accuracy - Testing data)  | 98.40% |
| **Rappel (Recall - Pointe)**          | **0.00%** |

### Pourquoi le Rappel est-il de 0% ?
Malgré une exactitude (accuracy) très élevée - (98% de bonnes réponses), le modèle échoue à détecter le moindre événement de pointe. 
C'est un cas classique du **Paradoxe de l'Exactitude**, causé par le déséquilibre des classes. C'est-à-dire que
les événements de pointe sont extrêmement rares dans nos données d'entraînement (environ 0.9 %). Ceci veut dire mathématiquement, 
la contribution des exemples "normaux" à la perte totale est plus grande comparée à celle des quelques exemples de "pointe". 
Ceci signifie que le moyen le plus rapide et le plus sûr pour le modèle de réduire cette somme est de toujours parier sur la majorité. 
Donc, prédire "Normal" tout le temps lui garantit une note de 99/100. Essayer de deviner une pointe est risqué: s'il se trompe, il augmente sa perte. 
Le modèle a donc appris que "ne rien faire" est la stratégie optimale pour minimiser l'erreur mathématique, même si cela contredit 
notre objectif d'affaires.


On peut employer 2 solutions pour forcer le modèle à prendre des risques: 
1. Ajuster le Seuil de Décision (Threshold Moving): Par défaut, la régression logistique classe en "pointe" si la probabilité $\hat{y} > 0.5$. 
Or, pour un événement rare, nous devrions être plus alertes. En abaissant ce seuil à 0.05, nous dirions au modèle : 
"Dès que tu perçois un risque de seulement 5 %, déclenche l'alarme". Cela augmenterait le rappel 
(détection). 
2. Pondérer la Fonction de Perte: Nous pouvons modifier l'équation de perte pour donner plus de "poids" aux erreurs sur les pointes.
$$J(\theta) = - \frac{1}{N} \sum [ \mathbf{100} \cdot y \log(\hat{y}) + (1-y) \log(1-\hat{y}) ]$$
Ici, rater une pointe coûterait 100 fois plus cher mathématiquement que de rater un moment normal. Le gradient forcerait 
le modèle à prêter attention à la classe minoritaire.

### Vers la Partie 3: L'importance des Données 
Avant de toucher aux poids, il faut se poser une question fondamentale : Le modèle a-t-il assez d'informations ?
Pour l'instant, notre modèle ne connait que la température actuelle. Mais une consommation élevée dépend souvent du contexte temporel :
- Fait-il froid depuis plusieurs heures ? (Inertie thermique)
- Est-ce le matin ou le soir ? (Habitudes de vie)
- La température a-t-elle chuté brutalement ?

C'est là qu'intervient **l'Ingénierie des Caractéristiques (Feature Engineering)**. Dans la prochaine partie, 
nous allons créer des variables temporelles (retards, moyennes glissantes) pour donner au modèle le contexte nécessaire 
pour distinguer une vraie pointe d'une simple journée froide.

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
