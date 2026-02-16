import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
from p3 import creer_caracteristiques
from p3_ami import creer_caracteristiques_ami_v2
from p3_ami import creer_caracteristiques_ami_v1

train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test_avec_cible.csv')
train_eng = creer_caracteristiques_ami_v1(train)
test_eng = creer_caracteristiques_ami_v1(test)

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
    # 'cycle_clients',
    'degre_jour_chauffage',
    'degre_jour_clim',
    'indice_temp_cons',
    'indice_temp_client',
    'facteur_eolien',
    # 'poste_A',
    # 'poste_B',
    # 'poste_C',
]

# Ajout dynamique des colonnes 'poste' (poste_A, poste_B, etc.)
features_postes = [c for c in train_eng.columns if c.startswith('poste_')]

# Liste finale complète
features_reg = features_base + features_eng + features_postes

# Vérification de sécurité
features_disponibles = [f for f in features_reg if f in train_eng.columns]
print(f"Caractéristiques utilisées ({len(features_disponibles)}) :")
print(features_disponibles)

X_train_raw = train_eng[features_disponibles].values
y_train = train_eng['energie_kwh'].values
X_test_raw = test_eng[features_disponibles].values
y_test = test_eng['energie_kwh'].values


# scaling avant
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_raw)
X_test_scaled = scaler.transform(X_test_raw)

# Modèle OLS (baseline)
print("\n--- Entraînement OLS ---")
model_ols = LinearRegression()
model_ols.fit(X_train_scaled, y_train)
y_pred_ols = model_ols.predict(X_test_scaled)

print("OLS (baseline):")
print(f"OLS  R² train: {model_ols.score(X_train_scaled, y_train):.4f}")
print(f"OLS  R² test:  {r2_score(y_test, y_pred_ols):.4f}")
print(f"OLS  RMSE test: {np.sqrt(mean_squared_error(y_test, y_pred_ols)):.4f}")

print("\n--- Entraînement Ridge (avec TimeSeriesSplit) ---")
# Modèle Ridge avec validation croisée
# ATTENTION: Utilisez TimeSeriesSplit pour les données temporelles!
from sklearn.model_selection import TimeSeriesSplit

alphas = [0.01, 0.1, 1, 10, 100, 1000]
tscv = TimeSeriesSplit(n_splits=5)

model_ridge = RidgeCV(alphas=alphas, cv=tscv)
model_ridge.fit(X_train_scaled, y_train)
y_pred_ridge = model_ridge.predict(X_test_scaled)

print(f"\nRidge (λ={model_ridge.alpha_}):")
print(f"Ridge  R² train: {model_ridge.score(X_train_scaled, y_train):.4f}")
print(f"Ridge  R² test:  {r2_score(y_test, y_pred_ridge):.4f}")
print(f"Ridge  RMSE test: {np.sqrt(mean_squared_error(y_test, y_pred_ridge)):.4f}")

# Comparaison des coefficients OLS vs Ridge
coef_comparison = pd.DataFrame({
    'Caractéristique': features_disponibles,
    'OLS': model_ols.coef_,
    'Ridge': model_ridge.coef_
})
coef_comparison['Réduction (%)'] = 100 * (1 - np.abs(coef_comparison['Ridge']) / (np.abs(coef_comparison['OLS']) + 1e-9))
coef_comparison = coef_comparison.sort_values('Réduction (%)', ascending=False)

# Affichage formaté
pd.options.display.float_format = '{:.4f}'.format

print("\n--- Comparaison des coefficients (Top 10 réductions) ---")
print(coef_comparison.head(10).to_string(index=False))

print("\n--- Top 5 des coefficients les plus importants (Ridge - Valeur Absolue) ---")
coef_comparison['Abs_Ridge'] = np.abs(coef_comparison['Ridge'])
print(coef_comparison.sort_values('Abs_Ridge', ascending=False)[['Caractéristique', 'Ridge']].head(5).to_string(index=False))

# --- Visualisation Graphique pour le Blogue ---
plt.figure(figsize=(12, 6))

top_indices = np.argsort(np.abs(model_ols.coef_))[-10:]
top_features = np.array(features_disponibles)[top_indices]

val_ols = model_ols.coef_[top_indices]
val_ridge = model_ridge.coef_[top_indices]

x = np.arange(len(top_features))
width = 0.35

plt.bar(x - width/2, val_ols, width, label='OLS (Standard)', color='salmon', alpha=0.7)
plt.bar(x + width/2, val_ridge, width, label='Ridge (Régularisé)', color='skyblue')

plt.xlabel('Caractéristiques')
plt.ylabel('Valeur du Coefficient')
plt.title('Effet de la Régularisation Ridge sur les Coefficients Instables')
plt.xticks(x, top_features, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
# plt.show()
# plt.savefig('partie4_coefficient.png')