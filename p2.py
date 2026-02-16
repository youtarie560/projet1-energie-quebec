import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings


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
    # 2. Initialiser beta à zéro
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


def plot():
    train = pd.read_csv('energy_train.csv')
    test = pd.read_csv('energy_test.csv')
    # Test sur la prédiction des événements de pointe
    # Caractéristiques pour classification
    features_clf = ['temperature_ext', 'heure_sin', 'heure_cos', 'est_weekend']

    X_train_clf = train[features_clf].values
    y_train_clf = train['evenement_pointe'].values
    X_test_clf = test[features_clf].values
    y_test_clf = test['evenement_pointe'].values

    # Normaliser (recommandé pour la descente de gradient)
    scaler = StandardScaler()
    X_train_clf_scaled = scaler.fit_transform(X_train_clf)
    X_test_clf_scaled = scaler.transform(X_test_clf)

    # Entraîner votre modèle
    beta_log, losses = logistic_fit_gd(X_train_clf_scaled, y_train_clf, lr=0.1, n_iter=500, verbose=True)

    # Tracer la courbe de convergence
    plt.figure(figsize=(8, 5))
    plt.plot(losses)
    plt.xlabel('Itération')
    plt.ylabel('Perte (entropie croisée)')
    plt.title('Convergence de la descente de gradient')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    # plt.savefig('convergence_plot.png')

    print()
    # Évaluation
    proba_train = logistic_predict_proba(X_train_clf_scaled, beta_log)
    proba_test = logistic_predict_proba(X_test_clf_scaled, beta_log)

    y_pred_train = (proba_train >= 0.5).astype(int)
    y_pred_test = (proba_test >= 0.5).astype(int)

    print("Évaluation de votre régression logistique:")
    print(f"  Accuracy (train): {accuracy_score(y_train_clf, y_pred_train):.4f}")
    print(f"  Accuracy (test): {accuracy_score(y_test_clf, y_pred_test):.4f}")
    print(f"\nRapport de classification (test):")
    print(classification_report(y_test_clf, y_pred_test, target_names=['Normal', 'Pointe']))

if __name__ == '__main__':
    plot()