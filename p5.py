import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from p3_ami import creer_caracteristiques_ami_v1

print("--- Chargement ---")
train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test_avec_cible.csv')

# Feature Engineering
train_eng = creer_caracteristiques_ami_v1(train).dropna()
test_eng = creer_caracteristiques_ami_v1(test).dropna()

# ==============================================================================
# 1. CLASSIFICATION (P_POINTE)
# ==============================================================================
print("\n--- 1. Entraînement du Classifieur ---")

features_pointe = ['temperature_ext', 'humidite', 'vitesse_vent', 'heure_sin', 'heure_cos', 'est_weekend',
                   'clients_connectes']
# Sécurité : on ne garde que les colonnes qui existent vraiment
features_pointe = [f for f in features_pointe if f in train_eng.columns]

X_train_clf = train_eng[features_pointe].values
y_train_clf = train_eng['evenement_pointe'].values
X_test_clf = test_eng[features_pointe].values
y_test_clf = test_eng['evenement_pointe'].values

# SCALING (Obligatoire)
scaler_clf = StandardScaler()
X_train_clf_scaled = scaler_clf.fit_transform(X_train_clf)
X_test_clf_scaled = scaler_clf.transform(X_test_clf)

# Entraînement
clf_pointe = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
clf_pointe.fit(X_train_clf_scaled, y_train_clf)

# Évaluation (SUR DONNÉES SCALÉES !)
# Correction : on utilise X_train_clf_scaled, pas X_train_clf
acc_train = clf_pointe.score(X_train_clf_scaled, y_train_clf)
acc_test = clf_pointe.score(X_test_clf_scaled, y_test_clf)
print(f"Accuracy (Train): {acc_train:.4f}")
print(f"Accuracy (Test):  {acc_test:.4f}")

# Extraction P(pointe)
train_eng['P_pointe'] = clf_pointe.predict_proba(X_train_clf_scaled)[:, 1]
test_eng['P_pointe'] = clf_pointe.predict_proba(X_test_clf_scaled)[:, 1]

print(f"Moyenne P(pointe) Train: {train_eng['P_pointe'].mean():.4f}")

# ==============================================================================
# 2. RÉGRESSION RIDGE FINALE
# ==============================================================================
print("\n--- 2. Entraînement Ridge Final ---")

# Liste cible de features
features_base = [
    'temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire',
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'jour_semaine_sin', 'jour_semaine_cos',
    'est_weekend', 'est_ferie', 'clients_connectes',
    'energie_lag1', 'energie_lag24',
    'degre_jour_chauffage', 'interaction_temp_vent',
    'indice_temp_cons', 'indice_temp_client', 'facteur_eolien',  # Ajout des autres vars physiques si dispos
    'P_pointe'
]

features_postes = [c for c in train_eng.columns if c.startswith('poste_')]
features_cibles = features_base + features_postes

# FILTRAGE FINAL : On ne garde que ce qui existe vraiment dans le DataFrame
features_disponibles = [f for f in features_cibles if f in train_eng.columns]
# Suppression des doublons potentiels
features_disponibles = list(dict.fromkeys(features_disponibles))

print(f"Nombre de features finales utilisées : {len(features_disponibles)}")

# Préparation
X_train_final = train_eng[features_disponibles].values
y_train_final = train_eng['energie_kwh'].values
X_test_final = test_eng[features_disponibles].values
y_test_final = test_eng['energie_kwh'].values

# Scaling Final
scaler_ridge = StandardScaler()
X_train_final_scaled = scaler_ridge.fit_transform(X_train_final)
X_test_final_scaled = scaler_ridge.transform(X_test_final)

# Entraînement
tscv = TimeSeriesSplit(n_splits=5)
alphas = [0.1, 1, 10, 100, 500]
model_final = RidgeCV(alphas=alphas, cv=tscv)
model_final.fit(X_train_final_scaled, y_train_final)

y_pred_final = model_final.predict(X_test_final_scaled)

# Résultats
r2_final = r2_score(y_test_final, y_pred_final)
rmse_final = np.sqrt(mean_squared_error(y_test_final, y_pred_final))
print(f"Ridge Final R² test:  {r2_final:.4f}")
print(f"Ridge Final RMSE:     {rmse_final:.4f}")

# ANALYSE DU COEFFICIENT P_POINTE (Correction de l'erreur d'index)
if 'P_pointe' in features_disponibles:
    # On cherche l'index dans la liste FILTRÉE (features_disponibles), pas la liste de base
    idx_p = features_disponibles.index('P_pointe')
    coef_p = model_final.coef_[idx_p]

    print(f"\nCoefficient de P_pointe : {coef_p:.4f}")
    if abs(coef_p) > 1:
        print("-> P_pointe a un impact significatif !")
    else:
        print("-> P_pointe a un impact faible.")
else:
    print("\nAttention : P_pointe n'a pas été utilisé (absent des features).")