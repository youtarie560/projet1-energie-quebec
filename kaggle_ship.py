import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from scipy.signal.windows import exponential

from p3_ami import creer_caracteristiques_ami_v1

train = pd.read_csv('energy_train.csv')
test_kaggle = pd.read_csv('energy_test.csv')

train_eng = creer_caracteristiques_ami_v1(train)
features_pointe = ['temperature_ext', 'humidite', 'vitesse_vent', 'heure_sin', 'heure_cos', 'est_weekend', 'clients_connectes']
features_pointe = [f for f in features_pointe if f in train_eng.columns]

scaler_clf = StandardScaler()
X_train_clf = scaler_clf.fit_transform(train_eng[features_pointe])
y_train_clf = train_eng['evenement_pointe'].values

clf_pointe = LogisticRegression(max_iter=1000, random_state=42)
clf_pointe.fit(X_train_clf, y_train_clf)


train_eng['P_pointe'] = clf_pointe.predict_proba(X_train_clf)[:, 1]

features_base = [
    'temperature_ext', 'humidite', 'vitesse_vent', 'irradiance_solaire',
    'heure_sin', 'heure_cos', 'mois_sin', 'mois_cos',
    'jour_semaine_sin', 'jour_semaine_cos',
    'est_weekend', 'est_ferie', 'clients_connectes'
]
features_eng = [
    'energie_lag1', 'energie_lag24', 'degre_jour_chauffage', 'degre_jour_clim',
    'indice_temp_cons', 'indice_temp_client', 'facteur_eolien', 'P_pointe'
]
if 'cycle_clients' in train_eng.columns: features_eng.append('cycle_clients')
features_postes = [c for c in train_eng.columns if c.startswith('poste_')]
features_final = features_base + features_eng + features_postes
features_final = list(dict.fromkeys([f for f in features_final if f in train_eng.columns]))
print(features_final)

scaler_ridge = StandardScaler()
X_train_reg = scaler_ridge.fit_transform(train_eng[features_final])
y_train_reg = train_eng['energie_kwh'].values

model_final = RidgeCV(alphas=[0.1, 1, 10, 100], cv=TimeSeriesSplit(n_splits=5))
model_final.fit(X_train_reg, y_train_reg)
print(f"Modèle Ridge entraîné (Alpha: {model_final.alpha_})")

test_kaggle['energie_kwh'] = np.nan

train['horodatage_local'] = pd.to_datetime(train['horodatage_local'])
test_kaggle['horodatage_local'] = pd.to_datetime(test_kaggle['horodatage_local'])

cutoff_date = train['horodatage_local'].max() - pd.Timedelta(days=7)
last_week_train = train[train['horodatage_local'] > cutoff_date].copy()

# On colle : [Fin du Train] + [Test Vide]
df_full = pd.concat([last_week_train, test_kaggle], axis=0, ignore_index=True)
df_full = df_full.sort_values(['poste', 'horodatage_local']).reset_index(drop=True)

dates_test = np.sort(test_kaggle['horodatage_local'].unique())

print(f"Début de la boucle sur {len(dates_test)} heures...")

for i, current_date in enumerate(dates_test):
    if i % 50 == 0:
        print(f"Traitement : {current_date} ({i}/{len(dates_test)})")

    df_features = creer_caracteristiques_ami_v1(df_full)

    mask = (df_features['horodatage_local'] == current_date) & (df_features['energie_kwh'].isna())

    if not mask.any():
        # mask = []
        continue

    X_clf_current = df_features.loc[mask, features_pointe].fillna(0)  # fillna sécurité météo

    X_clf_scaled = scaler_clf.transform(X_clf_current)

    # On injecte la proba dans le DataFrame temporaire
    p_pointe_vals = clf_pointe.predict_proba(X_clf_scaled)[:, 1]
    df_features.loc[mask, 'P_pointe'] = p_pointe_vals

    X_ridge_df = df_features.loc[mask, features_final]

    # Filet de sécurité : Si un lag est encore NaN (tout début), on remplit
    if X_ridge_df.isna().any().any():
        X_ridge_df = X_ridge_df.fillna(method='ffill').fillna(0)

    # Scaling (transform seulement !)
    X_ridge_scaled = scaler_ridge.transform(X_ridge_df)

    # Prédiction
    preds = model_final.predict(X_ridge_scaled)

    # ETAPE 4 : Mettre à jour le DataFrame Principal (df_full)
    # C'est ici qu'on "écrit l'histoire" pour le prochain tour de boucle
    indices_to_update = df_features.loc[mask].index
    df_full.loc[indices_to_update, 'energie_kwh'] = preds

submission_df = df_full[df_full['horodatage_local'].isin(dates_test)].copy()

final_output = pd.merge(test_kaggle[['horodatage_local', 'poste']],
                        submission_df[['horodatage_local', 'poste', 'energie_kwh']],
                        on=['horodatage_local', 'poste'],
                        how='left')

submission = pd.DataFrame({
    'id': range(len(final_output)),
    'energie_kwh': final_output['energie_kwh']
})

submission.to_csv('submission.csv', index=False)
