from meteostat import Point, Hourly
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import seaborn as sns

# Importation de votre fonction d'ingénierie existante
from p3_ami import creer_caracteristiques_ami_v1

def ajouter_meteo_externe(df_input):
    """
    Télécharge et fusionne les données de Pression et Point de Rosée via Meteostat.
    Station utilisée : Montréal-Trudeau (YUL) - ID 71627
    """
    df = df_input.copy()

    # 1. S'assurer que les dates sont au format datetime
    df['horodatage_local'] = pd.to_datetime(df['horodatage_local'])

    # CORRECTION 1 : Retrait brutal du fuseau horaire pour Meteostat
    # On force le mode "naïf" (sans info de fuseau) pour éviter le TypeError
    df['horodatage_local'] = df['horodatage_local'].dt.tz_localize(None)

    # 2. Définir la période à télécharger
    # .to_pydatetime() ne suffit parfois pas, on force replace(tzinfo=None)
    start = df['horodatage_local'].min().to_pydatetime().replace(tzinfo=None)
    end = df['horodatage_local'].max().to_pydatetime().replace(tzinfo=None)

    print(f"Téléchargement météo externe du {start} au {end}...")

    # 3. Définir le lieu (Montréal - YUL)
    location = Point(45.4690, -73.7447, 36)

    # 4. Récupérer les données horaires
    try:
        data = Hourly(location, start, end)
        data = data.fetch()
    except Exception as e:
        print(f"Erreur Meteostat : {e}")
        return df

    if data.empty:
        print("Attention: Aucune donnée météo récupérée.")
        return df

    # On ne garde que ce qui nous intéresse
    data_clean = data[['pres', 'dwpt']].copy()
    data_clean = data_clean.rename(columns={'pres': 'pression_atmos', 'dwpt': 'point_rosee'})

    # 5. Fusionner avec notre DataFrame
    data_clean['horodatage_local'] = data_clean.index
    # On s'assure que les données Meteostat sont aussi sans fuseau
    data_clean['horodatage_local'] = data_clean['horodatage_local'].dt.tz_localize(None)

    # Clé de fusion (String) pour éviter tout problème d'alignement
    df['merge_key'] = df['horodatage_local'].dt.strftime('%Y-%m-%d %H:%M:%S')
    data_clean['merge_key'] = data_clean['horodatage_local'].dt.strftime('%Y-%m-%d %H:%M:%S')

    df_merged = pd.merge(df, data_clean[['merge_key', 'pression_atmos', 'point_rosee']],
                         on='merge_key',
                         how='left')

    df_merged = df_merged.drop(columns=['merge_key'])

    # CORRECTION 2 : Remplacer les méthodes dépréciées (.fillna(method='...'))
    df_merged['pression_atmos'] = df_merged['pression_atmos'].interpolate(method='linear').bfill().ffill()
    df_merged['point_rosee'] = df_merged['point_rosee'].interpolate(method='linear').bfill().ffill()

    # Valeurs par défaut ultimes si échec
    df_merged['pression_atmos'] = df_merged['pression_atmos'].fillna(1013.25)
    df_merged['point_rosee'] = df_merged['point_rosee'].fillna(0)

    return df_merged

# ==============================================================================
# 1. CHARGEMENT ET PRÉPARATION
# ==============================================================================
print("--- Chargement des données ---")
train = pd.read_csv('energy_train.csv')
test = pd.read_csv('energy_test_avec_cible.csv')

print("--- Création des caractéristiques ---")
# .dropna() est vital ici !
train_eng = creer_caracteristiques_ami_v1(train).dropna()
test_eng = creer_caracteristiques_ami_v1(test).dropna()

print("--- Ajout Météo Externe (Option A) ---")
train_eng = ajouter_meteo_externe(train_eng)
test_eng = ajouter_meteo_externe(test_eng)

# ==============================================================================
# 2. DÉFINITION DES VARIABLES (FEATURES)
# ==============================================================================
# Liste de base
features_base = [
    'temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire',
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'jour_semaine_sin', 'jour_semaine_cos',
    'est_weekend', 'est_ferie', 'clients_connectes',
    'energie_lag1', 'energie_lag24',
    'degre_jour_chauffage', 'degre_jour_clim',
    'indice_temp_cons', 'indice_temp_client', 'facteur_eolien'
]

# Ajout dynamique
if 'cycle_clients' in train_eng.columns: features_base.append('cycle_clients')
features_postes = [c for c in train_eng.columns if c.startswith('poste_')]

# CORRECTION 3 : AJOUT EXPLICITE DES NOUVELLES VARIABLES
features_meteo_ext = ['pression_atmos', 'point_rosee']

# ==============================================================================
# 3. PARTIE 5 : CALCUL DE P(POINTE) (STACKING)
# ==============================================================================
print("\n--- Calcul de P(pointe) ---")
features_pointe = ['temperature_ext', 'humidite', 'vitesse_vent', 'heure_sin', 'heure_cos', 'est_weekend', 'clients_connectes']
features_pointe = [f for f in features_pointe if f in train_eng.columns]

scaler_clf = StandardScaler()
X_train_clf = scaler_clf.fit_transform(train_eng[features_pointe])
X_test_clf = scaler_clf.transform(test_eng[features_pointe])

y_train_clf = train_eng['evenement_pointe'].values
clf_pointe = LogisticRegression(max_iter=1000, random_state=42)
clf_pointe.fit(X_train_clf, y_train_clf)

train_eng['P_pointe'] = clf_pointe.predict_proba(X_train_clf)[:, 1]
test_eng['P_pointe'] = clf_pointe.predict_proba(X_test_clf)[:, 1]

# ==============================================================================
# 4. PARTIE 7 : MODÈLE FINAL ÉTENDU
# ==============================================================================
print("\n--- Modèle Final Étendu (Ridge + P_pointe + Météo Externe) ---")

# Construction de la liste finale AVEC la météo externe
features_final = features_base + features_postes + ['P_pointe'] + features_meteo_ext

# Filtrage de sécurité
features_final = [f for f in features_final if f in train_eng.columns]
print(f"Features utilisées ({len(features_final)}) : {features_final}")

X_train_final = train_eng[features_final].values
y_train_final = train_eng['energie_kwh'].values
X_test_final = test_eng[features_final].values
y_test_final = test_eng['energie_kwh'].values

# Scaling
scaler_final = StandardScaler()
X_train_final = scaler_final.fit_transform(X_train_final)
X_test_final = scaler_final.transform(X_test_final)

# Ridge
tscv = TimeSeriesSplit(n_splits=5)
model_final = RidgeCV(alphas=[0.1, 1, 10, 100], cv=tscv)
model_final.fit(X_train_final, y_train_final)
y_pred_final = model_final.predict(X_test_final)

# Résultats
print("\n=== RÉSULTATS PARTIE 7 (OPTION A) ===")
print(f"Ridge Étendu R² Test :  {r2_score(y_test_final, y_pred_final):.4f}")
print(f"Ridge Étendu RMSE Test: {np.sqrt(mean_squared_error(y_test_final, y_pred_final)):.4f}")

# Vérification des coefficients
print("\nImpact des nouvelles variables :")
for col in features_meteo_ext:
    if col in features_final:
        idx = features_final.index(col)
        print(f"  {col}: {model_final.coef_[idx]:.4f}")