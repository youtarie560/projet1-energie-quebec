import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings


def creer_caracteristiques(df: pd.DataFrame):
    """
    Version corrigée : Conversion Datetime + Physique du Collègue + Robustesse.
    """
    df = df.copy()

    df['horodatage_local'] = pd.to_datetime(df['horodatage_local'])
    df = df.sort_values(['poste', 'horodatage_local'])
    cols_meteo = ['temperature_ext', 'humidite', 'vitesse_vent']
    for c in cols_meteo:
        if c in df.columns:
            df[c] = df.groupby('poste')[c].ffill().bfill()

    # ==============================================================================
    # 2. GESTION DU LAG 1 (Ta méthode robuste validée)
    # ==============================================================================
    # --- PLAN A (ce quil sait passer il y a 1h) ---
    df_lookup = df[['horodatage_local', 'poste', 'energie_kwh']].copy()
    df_lookup_1h = df_lookup.copy()
    df_lookup_1h['horodatage_local'] = df_lookup_1h['horodatage_local'] + pd.Timedelta(hours=1)
    df_lookup_1h = df_lookup_1h.rename(columns={'energie_kwh': 'energie_lag1_raw'})

    df = pd.merge(df, df_lookup_1h, on=['horodatage_local', 'poste'], how='left')

    # --- PLAN B (MOYENNE MOBILE) ---
    def get_mean(group):
        g = group.set_index('horodatage_local')
        # closed='left' exclut l'heure actuelle
        # min_periods=1: suffit 1 donnée dans les 4h passées pour avoir un résultat
        rolling_res = g['energie_kwh'].rolling('4h', closed='left', min_periods=1).mean()
        return pd.Series(rolling_res.values, index=group.index)


    valeurs_de_secours = df.groupby('poste', group_keys=False)[['horodatage_local', 'energie_kwh']].apply(get_mean)
    if isinstance(valeurs_de_secours, pd.DataFrame):
        valeurs_de_secours = valeurs_de_secours.squeeze()

    valeurs_de_secours = valeurs_de_secours.sort_index()
    df['energie_lag1'] = df['energie_lag1_raw'].fillna(valeurs_de_secours)

    # --- PLAN C : LA DERNIÈRE CHANCE (FORWARD FILL) ---
    # Si on a encore des NaN (trous > 4h), on prend la consommation de la ligne PRÉCÉDENTE
    derniere_consommation_connue = df.groupby('poste')['energie_kwh'].shift(1)
    df['energie_lag1'] = df['energie_lag1'].fillna(derniere_consommation_connue)

    # --- DÉMARRAGE (BACKWARD FILL) ---
    #TODO: A REMPLACER. Pour la TOUTE première ligne du jeu de données (ex: 05h00) -
    df['energie_lag1'] = df.groupby('poste')['energie_lag1'].bfill()
    if 'energie_lag1_raw' in df.columns: df = df.drop(columns=['energie_lag1_raw'])

    # ==============================================================================
    # 3. GESTION DU LAG 24 (La mémoire journalière)
    # ==============================================================================
    # --- PLAN A (ce quil sait passer il y a 24h) ---
    df_lookup_24h = df_lookup.copy()
    df_lookup_24h['horodatage_local'] = df_lookup_24h['horodatage_local'] + pd.Timedelta(hours=24)
    df_lookup_24h = df_lookup_24h.rename(columns={'energie_kwh': 'energie_lag24_raw'})
    df = pd.merge(df, df_lookup_24h, on=['horodatage_local', 'poste'], how='left')

    # Plan B : Shift simple (Si l'heure exacte d'hier manque, prends la 24ème ligne avant)
    # C'est moins précis temporellement si des lignes manquent, mais ça sauve les meubles
    backup_24h = df.groupby('poste')['energie_kwh'].shift(24)
    df['energie_lag24'] = df['energie_lag24_raw'].fillna(backup_24h)
    # Plan C : Si on n'a toujours rien (début de dataset), on utilise le Lag 1 comme estimation
    # "Si je ne sais pas ce que j'ai consommé hier, je suppose que c'est comme il y a 1h"
    df['energie_lag24'] = df['energie_lag24'].fillna(df['energie_lag1'])
    if 'energie_lag24_raw' in df.columns: df = df.drop(columns=['energie_lag24_raw'])

    # ==============================================================================
    # 4. PHYSIQUE AVANCÉE (Les fondations solides)
    # Ces colonnes ne seront JAMAIS NaN car elles dépendent de la météo/temps
    # ==============================================================================
    df['cycle_clients'] = df['clients_connectes'] * (-df['heure_sin'])
    t_ext = df['temperature_ext'].fillna(0)
    df['degre_jour_chauffage'] = np.maximum(18 - t_ext, 0)
    df['degre_jour_clim'] = np.maximum(t_ext - 23, 0)

    # Indice Température Conso (Formule quadratique pondérée)
    # 2x pour le chauffage, 0.65x pour la clim
    df['indice_temp_cons'] = np.square(2 * df['degre_jour_chauffage'] + 0.65 * df['degre_jour_clim'])

    # Interaction Taille Réseau
    df['indice_temp_client'] = df['indice_temp_cons'] * df['clients_connectes']

    # Facteur Éolien (Vent * Indice Thermique)
    df['facteur_eolien'] = df['indice_temp_cons'] * df['vitesse_vent']

    if 'poste' in df.columns:
        df = pd.get_dummies(df, columns=['poste'], prefix='poste', dtype=int)

    return df


train = pd.read_csv('energy_train.csv')
train_eng = creer_caracteristiques(train)
# train_eng.to_csv('p3.csv', index=False)
print(f"Nouvelles colonnes: {[c for c in train_eng.columns if c not in train.columns]}")
# ['energie_lag1', 'energie_lag24', 'cycle_clients', 'degre_jour_chauffage', 'degre_jour_clim', 'indice_temp_cons', 'indice_temp_client', 'facteur_eolien', 'poste_A', 'poste_B', 'poste_C']
