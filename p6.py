import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Importation de votre fonction robuste (depuis le fichier p3_ami.py)
from p3_ami import creer_caracteristiques_ami_v1

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION
# ==============================================================================
print("--- Chargement des données ---")
train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test_avec_cible.csv')

print("--- Création des caractéristiques ---")
# On utilise la fonction qui gère bien les index et les dates
train_eng = creer_caracteristiques_ami_v1(train).dropna()
test_eng = creer_caracteristiques_ami_v1(test).dropna()

# ==============================================================================
# 2. DÉFINITION DES VARIABLES (FEATURES)
# ==============================================================================
features_base = [
    'temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire',
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'jour_semaine_sin', 'jour_semaine_cos',
    'est_weekend', 'est_ferie',
    'clients_connectes'
]

features_eng = [
    'energie_lag1',
    'energie_lag24',
    'degre_jour_chauffage',
    'degre_jour_clim',
    'indice_temp_cons',
    'indice_temp_client',
    'facteur_eolien'
]
if 'cycle_clients' in train_eng.columns:
    features_eng.append('cycle_clients')

# Ajout dynamique des postes
features_postes = [c for c in train_eng.columns if c.startswith('poste_')]

# Liste disponible (sans doublons)
features_disponibles = list(dict.fromkeys(
    [f for f in features_base + features_eng + features_postes if f in train_eng.columns]
))

# ==============================================================================
# 3. PARTIE 4 (POUR COMPARAISON) : MODÈLES DE BASE
# ==============================================================================
print("\n--- Ré-entraînement des bases (Partie 4) pour comparaison ---")
X_train = train_eng[features_disponibles].values
y_train = train_eng['energie_kwh'].values
X_test = test_eng[features_disponibles].values
y_test = test_eng['energie_kwh'].values

# Scaling (Indispensable)
scaler_base = StandardScaler()
X_train_scaled = scaler_base.fit_transform(X_train)
X_test_scaled = scaler_base.transform(X_test)

# OLS
model_ols = LinearRegression()
model_ols.fit(X_train_scaled, y_train)
y_pred_ols = model_ols.predict(X_test_scaled)

# Ridge Base
tscv = TimeSeriesSplit(n_splits=5)
model_ridge = RidgeCV(alphas=[0.1, 1, 10, 100], cv=tscv)
model_ridge.fit(X_train_scaled, y_train)
y_pred_ridge = model_ridge.predict(X_test_scaled)

# ==============================================================================
# 4. PARTIE 5 : CLASSIFIEUR DE POINTE
# ==============================================================================
print("\n--- Partie 5 : Calcul de P(pointe) ---")
# Features "légales" pour la classification (sans lags conso)
features_pointe = ['temperature_ext', 'humidite', 'vitesse_vent', 'heure_sin', 'heure_cos', 'est_weekend', 'clients_connectes']
features_pointe = [f for f in features_pointe if f in train_eng.columns]

scaler_clf = StandardScaler()
X_train_clf = scaler_clf.fit_transform(train_eng[features_pointe])
X_test_clf = scaler_clf.transform(test_eng[features_pointe])

y_train_clf = train_eng['evenement_pointe'].values
y_test_clf = test_eng['evenement_pointe'].values

# Entraînement
clf_pointe = LogisticRegression(max_iter=1000, random_state=42)
clf_pointe.fit(X_train_clf, y_train_clf)

# Ajout de la probabilité aux DataFrames
train_eng['P_pointe'] = clf_pointe.predict_proba(X_train_clf)[:, 1]
test_eng['P_pointe'] = clf_pointe.predict_proba(X_test_clf)[:, 1]

print(f"  Distribution P(pointe) Test : Moy={test_eng['P_pointe'].mean():.3f}")

# ==============================================================================
# 5. PARTIE 6 : MODÈLE FINAL ASSEMBLÉ
# ==============================================================================
print("\n--- Partie 6 : Modèle Final (Ridge + P_pointe) ---")

# Ajout de la nouvelle caractéristique
features_final = features_disponibles + ['P_pointe']

X_train_final_raw = train_eng[features_final].values
y_train_final = train_eng['energie_kwh'].values
X_test_final_raw = test_eng[features_final].values
y_test_final = test_eng['energie_kwh'].values

# Scaling Final (Important !)
scaler_final = StandardScaler()
X_train_final = scaler_final.fit_transform(X_train_final_raw)
X_test_final = scaler_final.transform(X_test_final_raw)

# Modèle Ridge Final
model_final = RidgeCV(alphas=[0.1, 1, 10, 100], cv=tscv)
model_final.fit(X_train_final, y_train_final)
y_pred_final = model_final.predict(X_test_final)

# Résultats
print("\nRésultats du Modèle Final :")
print(f"  λ sélectionné: {model_final.alpha_}")
print(f"  R² train: {model_final.score(X_train_final, y_train_final):.4f}")
print(f"  R² test:  {r2_score(y_test_final, y_pred_final):.4f}")
print(f"  RMSE test: {np.sqrt(mean_squared_error(y_test_final, y_pred_final)):.4f}")

# Comparaison
print("\n=== Récapitulatif ===")
r2_ols = r2_score(y_test, y_pred_ols)
r2_ridge = r2_score(y_test, y_pred_ridge)
r2_final = r2_score(y_test_final, y_pred_final)

print(f"OLS baseline:     R² = {r2_ols:.4f}")
print(f"Ridge:            R² = {r2_ridge:.4f}")
print(f"Ridge + P_pointe: R² = {r2_final:.4f}")

gain = 100 * (r2_final - r2_ridge)
print(f"\nAmélioration due à P_pointe: {gain:+.2f}%")

# ==============================================================================
# 6. VISUALISATION
# ==============================================================================
residus = y_test_final - y_pred_final
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogramme des résidus
axes[0].hist(residus, bins=50, edgecolor='black', alpha=0.7, color='skyblue')
axes[0].axvline(0, color='red', linestyle='--', label='Zéro')
axes[0].set_xlabel('Résidu (Erreur en kWh)')
axes[0].set_ylabel('Fréquence')
axes[0].set_title('Distribution des résidus')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# Prédictions vs Réel
axes[1].scatter(y_test_final, y_pred_final, alpha=0.3, s=10, color='purple')
axes[1].plot([y_test_final.min(), y_test_final.max()],
             [y_test_final.min(), y_test_final.max()], 'r--', label='Parfait')
axes[1].set_xlabel('Énergie réelle (kWh)')
axes[1].set_ylabel('Énergie prédite (kWh)')
axes[1].set_title('Prédictions vs Réel')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()