"""Point d'entrée : exécute les étapes du pipeline via des arguments CLI.

Exemples :
    python main.py --prepare
    python main.py --train
    python main.py --evaluate
    python main.py --prepare --train --evaluate
"""

import argparse

from model_pipeline import (
    evaluate_model,
    load_model,
    prepare_data,
    save_model,
    train_model,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Pipeline ML - Churn")
    parser.add_argument("--data", default="Churn_Modelling.csv",
                        help="chemin du fichier CSV")
    parser.add_argument("--model-path", default="classifier.joblib",
                        help="chemin du modèle sauvegardé")
    parser.add_argument("--prepare", action="store_true",
                        help="charger et prétraiter les données")
    parser.add_argument("--train", action="store_true",
                        help="entraîner puis sauvegarder le modèle")
    parser.add_argument("--evaluate", action="store_true",
                        help="charger le modèle et l'évaluer")
    return parser.parse_args()


def main():
    args = parse_args()
    if not (args.prepare or args.train or args.evaluate):
        print("Aucune action demandée. Utilisez --prepare, --train ou --evaluate.")
        return

    # Les données sont préparées pour toutes les étapes (découpage reproductible)
    x_train, x_test, y_train, y_test = prepare_data(args.data)
    if args.prepare:
        print(f"Données prêtes : train={x_train.shape}, test={x_test.shape}")

    if args.train:
        model = train_model(x_train, y_train)
        save_model(model, args.model_path)

    if args.evaluate:
        model = load_model(args.model_path)
        results = evaluate_model(model, x_test, y_test)
        print(f"Accuracy : {results['accuracy'] * 100:.2f} %")
        print("Matrice de confusion :")
        print(results["confusion_matrix"])
        print(results["report"])


if __name__ == "__main__":
    main()
