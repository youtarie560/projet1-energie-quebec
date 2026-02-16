import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')


def ols_fit(X, y):
    """
    Calcule les coefficients OLS.

    Paramètres:
        X : ndarray de forme (n, p) - matrice de caractéristiques (SANS colonne de 1)
                            n-> #of rows, p-># of features
        y : ndarray de forme (n,) - vecteur cible

    Retourne:
        beta : ndarray de forme (p+1,) - coefficients [intercept, coef1, coef2, ...]

    Indice: Ajoutez une colonne de 1 à X pour l'intercept.
    """
    # VOTRE CODE ICI
    # 1. Ajouter une colonne de 1 pour l'intercept (le bias)
    #  columne de 1 de taille (nx1)
    ones_col = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones_col, X]) # size (n, p+1)

    # 2. Résoudre le système X^T X beta = X^T y
    # Terme de gauche : X^T * X
    terme_gauche = X_aug.T @ X_aug # matrice carrée symétrique de taille (p+1xp+1)

    # Terme de droite : X^T * y
    terme_droite = X_aug.T @ y # vecteur de taille (p+1)

    try:
        beta = np.linalg.solve(terme_gauche, terme_droite)
    except np.linalg.LinAlgError:
        # si la matrice est singulière (non inversible), utilise les moindres carrés via pseudo-inverse (SVD)
        print("Attention: Matrice singulière détectée, utilisation de lstsq.")
        beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
    # 3. Retourner beta
    return beta


def ols_predict(X, beta):
    """
    Prédit avec les coefficients OLS.

    Paramètres:
        X : ndarray de forme (n, p) - caractéristiques (SANS colonne de 1)
        beta : ndarray de forme (p+1,) - coefficients [intercept, coef1, ...]

    Retourne:
        y_pred : ndarray de forme (n,)
    """
    # VOTRE CODE ICI
    ones_col = np.ones((X.shape[0], 1))
    X_aug = np.hstack([ones_col, X])

    return X_aug @ beta

def plot():
    # Test de votre implémentation
    # Caractéristiques simples pour commencer
    train = pd.read_csv('energy_train.csv')
    test = pd.read_csv('energy_test.csv')
    features_base = ['temperature_ext', 'humidite', 'vitesse_vent']

    X_train_base = train[features_base].values
    y_train = train['energie_kwh'].values
    X_test_base = test[features_base].values
    y_test = test['energie_kwh'].values

    # notre implémentation
    beta_ols = ols_fit(X_train_base, y_train)
    y_pred_ols = ols_predict(X_test_base, beta_ols)

    # Validation avec sklearn
    model_sklearn = LinearRegression()
    model_sklearn.fit(X_train_base, y_train)
    y_pred_sklearn = model_sklearn.predict(X_test_base)

    # Comparaison
    print("Comparaison OLS implémenté vs sklearn:")
    print(f"  Intercept - Vous: {beta_ols[0]:.4f}, sklearn: {model_sklearn.intercept_:.4f}")
    print(f"  Coefficients proches: {np.allclose(beta_ols[1:], model_sklearn.coef_, atol=1e-4)}")
    print(f"\nR² sur test: {r2_score(y_test, y_pred_ols):.4f}")
    print(f"\nR² sur test (de sklearn.linear_model.LinearRegression()): {r2_score(y_test, y_pred_sklearn):.4f}")


plot()