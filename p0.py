import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, RidgeCV, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Configuration des graphiques
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12
plt.rcParams['figure.dpi'] = 200

print("Configuration terminée!")

# URLs des données sur GitHub
BASE_URL = "https://raw.githubusercontent.com/pierrelux/mlbook/main/data/"
TRAIN_DATA = 'energy_train.csv'
TEST_DATA = 'energy_train.csv'
TEST_DATA_KAGGLE = 'energy_train.csv'
# Charger les données
train = pd.read_csv(BASE_URL + "energy_train.csv", parse_dates=['horodatage_local'])
train.to_csv("energy_train.csv", index=False)
# Pour l'évaluation locale: test avec la cible (energie_kwh)
test = pd.read_csv(BASE_URL + "energy_test_avec_cible.csv", parse_dates=['horodatage_local'])
test.to_csv("energy_test_avec_cible.csv", index=False)
# Pour Kaggle: test sans la cible (pour générer les prédictions)
test_kaggle = pd.read_csv(BASE_URL + "energy_test.csv", parse_dates=['horodatage_local'])
test_kaggle.to_csv("energy_test.csv", index=False)
train_df = pd.read_csv(TRAIN_DATA, parse_dates=['horodatage_local'])
test_df = pd.read_csv(TRAIN_DATA, parse_dates=['horodatage_local'])
test_kaggle_df = pd.read_csv(TRAIN_DATA, parse_dates=['horodatage_local'])

print(f"Ensemble d'entraînement: {len(train_df)} observations")
print(f"Ensemble de test: {len(test_df)} observations")
# print(f"\nPériode d'entraînement: {train_df['horodatage_local'].min()} à {train_df['horodatage_local'].max()}")
# print(f"Période de test: {test_df['horodatage_local'].min()} à {test_df['horodatage_local'].max()}")