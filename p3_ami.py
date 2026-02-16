import numpy as np
import pandas as pd
from scipy.signal.windows import exponential


# Fonction helper (optimisée pour NumPy)
def exp_mean_numpy(window_data, tau=2, symmetry=True):
    n = len(window_data)
    if n == 0:
        return np.nan
    weights = exponential(n, tau=tau, sym=symmetry)
    return np.average(window_data, weights=weights)


def exp_mean(window_data, tau, symmetry):
    # Nombre de points dans la fenêtre
    n = len(window_data)

    if n == 0:
        return np.nan

    # Génère une distribution exponentielle avec scipy.signal.exponential
    # sym détermine si on veut une symmétrie autour du centre
    # tau détermine la vitesse du decay
    weights = exponential(n, tau=tau, sym=symmetry)

    # Normalise le poids
    weights = weights / np.sum(weights)

    # Calcule la moyenne pondérée
    weighted_avg = np.sum(window_data * weights) / np.sum(weights)
    return weighted_avg

def creer_caracteristiques_ami_v2(df: pd.DataFrame):
    """
    Crée des caractéristiques supplémentaires (Version Corrigée & Robuste).
    """
    df = df.copy()

    # 1. Conversion Datetime
    df['horodatage_local'] = pd.to_datetime(df['horodatage_local'])

    # 2. Tri CRUCIAL : Garantit que l'assignation .values plus bas sera alignée
    df = df.sort_values(['poste', 'horodatage_local'])

    # --- FEATURE ENGINEERING ---

    # A. Retards (Lags)
    # Groupby assure qu'on ne décale pas les données du Poste A vers le Poste B
    # shift(1) préserve l'index original, donc l'assignation est sûre.
    df['energie_lag1'] = df.groupby('poste')['energie_kwh'].shift(1).fillna(0)

    # B. Lag 24h (Moyenne Exponentielle)
    # Calcul du rolling
    rolling_result = df.groupby('poste').rolling(
        window="48h",  # 'h' minuscule pour éviter le warning
        on='horodatage_local',
        closed='left'
    )['energie_kwh'].apply(
        lambda x: exp_mean_numpy(x, tau=2, symmetry=True),
        raw=True
    )

    # CORRECTION ICI : On assigne les .values directement
    # Cela contourne le problème d'alignement d'index (ValueError duplicate labels)
    # C'est sûr car df est trié par ['poste', 'horodatage_local'], tout comme le résultat du groupby.
    df['energie_lag24'] = rolling_result.values
    df['energie_lag24'] = df['energie_lag24'].fillna(0)

    # C. Physique & Interactions
    df['degre_jour_chauffage'] = np.maximum(18 - df['temperature_ext'], 0)
    df['degre_jour_clim'] = np.maximum(df['temperature_ext'] - 23, 0)

    df['indice_temp_cons'] = np.square(2 * df['degre_jour_chauffage'] + 0.65 * df['degre_jour_clim'])
    df['indice_temp_client'] = df['indice_temp_cons'] * df['clients_connectes']

    df['facteur_eolien'] = df['indice_temp_cons'] * df['vitesse_vent']

    # Cycle Client
    if 'heure' not in df.columns:
        df['heure'] = df['horodatage_local'].dt.hour

    # Formule cycle client
    df['cycle_clients'] = df['clients_connectes'] * (-np.sin(df['heure'] * (np.pi) / 24))

    return df


def creer_caracteristiques_ami_v1(df: pd.DataFrame):
    """
    Crée des caractéristiques supplémentaires.

    VOUS DEVEZ IMPLÉMENTER AU MOINS 3 NOUVELLES CARACTÉRISTIQUES.

    Idées:
    - Retards: df['energie_kwh'].shift(1), shift(24)
    - Moyennes mobiles: df['energie_kwh'].rolling(6).mean()
    - Interactions: df['temperature_ext'] * df['heure_cos']
    - Degré-jours de chauffage: np.maximum(18 - df['temperature_ext'], 0)
    """
    df = df.copy()
    df['horodatage_local'] = pd.to_datetime(df['horodatage_local'])

    # Séparation des 'poste' (poste_A, poste_B, etc.)
    postes = df['poste'].unique()
    postes = sorted(postes)
    # print(postes)
    df = [df[(df['poste'] == postes[i])] for i in range(len(postes))]

    # print("Nombre d'entrées par poste: ")
    # for i in range(len(postes)):
        # print(f"{postes[i]}: {len(df[i])}")

    for i in range(len(postes)):
        # 1. Retards (Lags) - La mémoire du système
        # Ce qu'on consommait au dernier échantillonage à ce poste
        df[i]['energie_lag1'] = df[i]['energie_kwh'].shift(1)

        df[i]['energie_lag1'] = df[i]['energie_lag1'].fillna(0)
        # Ce qu'on consommait il y a 24 heures (Cycle d'activité humaine)
        # On prend une distribution exponentielle symmétrique sur la fenêtre des 24 heures strictement avant la date ciblée
        df[i]['energie_lag24'] = \
        df[i][['horodatage_local', 'energie_kwh']].rolling(window="48h", on='horodatage_local', closed='left').apply(
            lambda rows: exp_mean(rows, tau=2, symmetry=True), raw=False)['energie_kwh']
        df[i]['energie_lag24'] = df[i]['energie_lag24'].fillna(0)

        # Moyenne des 6 dernières heures pour lisser le bruit
        # df[i]['energie_rolling_mean_6h'] = df[i][['horodatage_local','energie_kwh']].rolling(window="12H", on='horodatage_local', closed='left').apply(lambda x: exp_mean(x, tau=2, symmetry=False), raw=False)['energie_kwh']
        # df[i]['energie_rolling_mean_6h'] = df[i]['energie_rolling_mean_6h'].fillna(0)

    # for i in range(len(postes)):
    # df[i]['mean_6h_per_client'] = df[i]['energie_rolling_mean_6h']/df[i]['clients_connectes']*(-np.sin(df[i]['heure']*(np.pi)/24))

    # Moyenne mobile
    # for i in range(len(postes)):
    # df[i]['mean_6h_cycle'] = df[i]['energie_rolling_mean_6h']*(-np.sin(df[i]['heure']*(np.pi)/24))

    # print(poste_A['energie_rolling_mean_6h'])

    df = pd.concat(df)
    df.sort_values(by='horodatage_local', inplace=True)
    # print(df)
    # 3. Interactions Physiques - La réalité du chauffage
    # Degrés-Jours de Chauffage (HDD) : On ne chauffe que si T < 18°C.
    # C'est une transformation non-linéaire cruciale : la relation Conso vs Temp est plate au-dessus de 18°C.
    df['degre_jour_chauffage'] = np.maximum(18 - df['temperature_ext'], 0)

    df['degre_jour_clim'] = np.maximum(df['temperature_ext'] - 23, 0)

    # Température intérieure saisonière variable
    # On assume qu'en général, la températures des logis sera correlée à la température extérieure dans un certain interval.
    # df['int_temp_estimate'] = np.clip(df['temperature_ext'],18,23)

    # Interactions physiques - Indice de consommation pour la température
    # Prend en compte la consommation plus élevée d'électricité pour chauffer que pour refroidir
    df['indice_temp_cons'] = np.square(2 * df['degre_jour_chauffage'] + (0.65) * df['degre_jour_clim'])
    df['indice_temp_client'] = df['indice_temp_cons'] * df['clients_connectes']
    # Interactions physiques - Transfert de chaleur à l'environnement
    # Le transfert de chaleur entre l'extérieur et l'intérieur peut être capturé par le carré de la différence de température entre les milieux.
    # df['temp_diff_squared'] = np.square(np.clip(df['temperature_ext'],22,22)-df['temperature_ext'])
    # df['temp_diff_squared'] = np.abs(np.clip(df['temperature_ext'],20,24)-df['temperature_ext'])

    # Facteur d'augmentation du transfert de chaleur (facteur éolien)
    # Le vent augmente l'échange de chaleur entre les milieux
    df['facteur_eolien'] = df['indice_temp_cons'] * df['vitesse_vent']
    df.to_csv(f"output{len(df)}.csv")
    return df

def run_p3():
    train = pd.read_csv('energy_train.csv')
    # test = pd.read_csv('energy_test.csv')

    print("Création des caractéristiques...")
    train_eng = creer_caracteristiques_ami_v1(train)

    # Nettoyage des NaNs résiduels si nécessaire (ex: début de série)
    train_eng = train_eng.dropna()

    cols_nouvelles = [c for c in train_eng.columns if c not in train.columns]
    print(f"Nouvelles colonnes ({len(cols_nouvelles)}): {cols_nouvelles}")

    return train_eng


if __name__ == '__main__':
    run_p3()