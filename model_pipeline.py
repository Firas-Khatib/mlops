"""Pipeline ML modulaire pour la prédiction du churn (Churn_Modelling.csv).

Fonctions issues de la modularisation du notebook customer_churn.ipynb :
prepare_data, train_model, evaluate_model, save_model, load_model.
"""

import os

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
)
from sklearn.model_selection import train_test_split

TARGET = "Exited"
# Colonnes non informatives ou non utilisées dans le notebook
COLUMNS_TO_DROP = ["RowNumber", "CustomerId", "Surname", "Geography"]


def prepare_data(csv_path, test_size=0.2, random_state=1):
    """Charge le CSV et prétraite les données.

    Étapes : vérification du fichier, suppression des doublons, encodage de
    Gender (Female=0, Male=1), suppression des colonnes inutiles, séparation
    variables/cible puis découpage train/test.

    Args:
        csv_path (str): chemin du fichier CSV.
        test_size (float): proportion du jeu de test (entre 0 et 1).
        random_state (int): graine pour un découpage reproductible.

    Returns:
        tuple: (x_train, x_test, y_train, y_test)
    """
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")
    if not 0 < test_size < 1:
        raise ValueError("test_size doit être compris entre 0 et 1.")

    df = pd.read_csv(csv_path)
    if TARGET not in df.columns:
        raise ValueError(f"Colonne cible '{TARGET}' absente du fichier.")

    df = df.drop_duplicates()

    # Même résultat que LabelEncoder (ordre alphabétique), mais déterministe
    df["Gender"] = df["Gender"].map({"Female": 0, "Male": 1})

    df = df.drop(columns=[c for c in COLUMNS_TO_DROP if c in df.columns])

    x = df.drop(columns=[TARGET])
    y = df[TARGET]

    return train_test_split(x, y, test_size=test_size, random_state=random_state)


def train_model(x_train, y_train, n_estimators=100, random_state=42):
    """Entraîne un RandomForestClassifier.

    Args:
        x_train (DataFrame): variables explicatives d'entraînement.
        y_train (Series): cible d'entraînement.
        n_estimators (int): nombre d'arbres.
        random_state (int): graine aléatoire.

    Returns:
        RandomForestClassifier: modèle entraîné.
    """
    if len(x_train) != len(y_train):
        raise ValueError("x_train et y_train doivent avoir la même longueur.")

    model = RandomForestClassifier(
        n_estimators=n_estimators, random_state=random_state
    )
    model.fit(x_train, y_train)
    return model


def evaluate_model(model, x_test, y_test):
    """Évalue le modèle sur le jeu de test.

    Args:
        model: modèle entraîné (méthode predict).
        x_test (DataFrame): variables explicatives de test.
        y_test (Series): cible de test.

    Returns:
        dict: accuracy, matrice de confusion et rapport de classification.
    """
    y_pred = model.predict(x_test)
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "report": classification_report(y_test, y_pred),
    }


def save_model(model, path="classifier.joblib"):
    """Sauvegarde le modèle avec joblib.

    Args:
        model: modèle entraîné.
        path (str): chemin du fichier de sortie.
    """
    joblib.dump(model, path)
    print(f"Modèle sauvegardé : {path}")


def load_model(path="classifier.joblib"):
    """Charge un modèle sauvegardé avec joblib.

    Args:
        path (str): chemin du fichier du modèle.

    Returns:
        Le modèle chargé.
    """
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"Modèle introuvable : {path}. Lancez d'abord --train."
        )
    return joblib.load(path)
