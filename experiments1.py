# Cleaned experiments.py
from modules.preprocess import preprocessing, split
from modules.evaluate import evaluate_performance
from modules.print_draw import print_data, draw_loss
from models.models import create_nn_model, train_model, model_predict

import pandas as pd
import joblib
from os.path import join as join

import mlflow
import mlflow.tensorflow

NUM_COLS = ["age", "taille", "poids", "revenu_estime_mois"]
CAT_COLS = ["sexe", "sport_licence", "niveau_etude", "region", "smoker", "nationalité_francaise"]
TARGET_COL = "montant_pret"
DROP_COLS = ["nom", "prenom", TARGET_COL]

def load_and_preprocess_with_saved_preprocessor(csv_path, preprocessor):
    df = pd.read_csv(csv_path)
    X_raw = df.drop(columns=DROP_COLS)
    y = df[TARGET_COL]
    X_processed = preprocessor.transform(X_raw)
    return X_processed, y

def exp1_baseline_old_model_on_df_old():
    print("\n=== Expérience 1 : Ancien modèle sur df_old (baseline) ===")
    df_old = pd.read_csv(join("data", "df_old.csv"))
    X_old, y_old, preprocessor_old = preprocessing(df_old)
    X_train_old, X_test_old, y_train_old, y_test_old = split(X_old, y_old)
    model_2024_08 = joblib.load(join("models", "model_2024_08.pkl"))

    mlflow.set_experiment("OPCO_M1_Brief1")
    with mlflow.start_run(run_name="Exp1_old_model_df_old"):
        mlflow.log_param("exp", "1")
        y_pred_train = model_predict(model_2024_08, X_train_old)
        perf_train = evaluate_performance(y_train_old, y_pred_train)
        print_data(perf_train, exp_name="Exp 1 - train")

        y_pred_test = model_predict(model_2024_08, X_test_old)
        perf_test = evaluate_performance(y_test_old, y_pred_test)
        print_data(perf_test, exp_name="Exp 1 - test")

        mlflow.log_metrics({k: perf_train.get(k) for k in perf_train})
        mlflow.log_metrics({"test_" + k: perf_test.get(k) for k in perf_test})

    return model_2024_08, preprocessor_old, X_train_old, X_test_old, y_train_old, y_test_old

def exp2_old_model_on_df_new(model, preprocessor):
    print("\n=== Expérience 2 : Ancien modèle sur df_new ===")
    X_new, y_new = load_and_preprocess_with_saved_preprocessor(join("data","df_new.csv"), preprocessor)
    X_train_new, X_test_new, y_train_new, y_test_new = split(X_new, y_new)

    mlflow.set_experiment("OPCO_M1_Brief1")
    with mlflow.start_run(run_name="Exp2_old_model_df_new"):
        mlflow.log_param("exp", "2")
        y_pred_new = model_predict(model, X_test_new)
        perf_new = evaluate_performance(y_test_new, y_pred_new)
        print_data(perf_new, exp_name="Exp 2 - old model sur df_new")

        mlflow.log_metrics({k: perf_new.get(k) for k in perf_new})

    return X_train_new, X_test_new, y_train_new, y_test_new

def exp3_retrain_old_model_on_df_old(model, X_train_old, y_train_old, X_test_old, y_test_old):
    print("\n=== Expérience 3 : Ré-entrainer l'ancien modèle sur df_old ===")
    mlflow.set_experiment("OPCO_M1_Brief1")
    with mlflow.start_run(run_name="Exp3_retrain_old_model_df_old"):
        mlflow.log_param("exp", "3")
        model2, hist2 = train_model(model, X_train_old, y_train_old, X_val=X_test_old, y_val=y_test_old, epochs=10, batch_size=32, verbose=0)

        y_pred_test = model_predict(model2, X_test_old)
        perf_test = evaluate_performance(y_test_old, y_pred_test)
        print_data(perf_test, exp_name="Exp 3 - old model réentraîné sur df_old")

        mlflow.log_metrics({k: perf_test.get(k) for k in perf_test})

    draw_loss(hist2)
    return model2

def exp4_retrain_old_model_on_df_new(model, X_train_new, y_train_new, X_test_new, y_test_new):
    print("\n=== Expérience 4 : Ré-entrainer l'ancien modèle sur df_new ===")
    mlflow.set_experiment("OPCO_M1_Brief1")
    with mlflow.start_run(run_name="Exp4_retrain_old_model_df_new"):
        mlflow.log_param("exp", "4")
        model3, hist3 = train_model(model, X_train_new, y_train_new, X_val=X_test_new, y_val=y_test_new, epochs=10, batch_size=32, verbose=0)

        y_pred_test = model_predict(model3, X_test_new)
        perf_test = evaluate_performance(y_test_new, y_pred_test)
        print_data(perf_test, exp_name="Exp 4 - old model réentraîné sur df_new")

        mlflow.log_metrics({k: perf_test.get(k) for k in perf_test})

    draw_loss(hist3)
    return model3

def exp5_new_model_on_df_new(X_train_new, y_train_new, X_test_new, y_test_new):
    print("\n=== Expérience 5 : Nouveau modèle sur df_new ===")
    mlflow.set_experiment("OPCO_M1_Brief1")
    with mlflow.start_run(run_name="Exp5_new_model_df_new"):
        mlflow.log_param("exp", "5")
        model_new = create_nn_model(X_train_new.shape[1])
        model_new, hist_new = train_model(model_new, X_train_new, y_train_new, X_val=X_test_new, y_val=y_test_new, epochs=20, batch_size=32, verbose=0)

        y_pred_test = model_predict(model_new, X_test_new)
        perf_test = evaluate_performance(y_test_new, y_pred_test)
        print_data(perf_test, exp_name="Exp 5 - new model sur df_new")

        mlflow.log_metrics({k: perf_test.get(k) for k in perf_test})

    draw_loss(hist_new)
    return model_new

def exp6_new_model_on_df_old(new_model, preprocessor):
    print("\n=== Expérience 6 : Nouveau modèle évalué sur df_old ===")
    X_old, y_old = load_and_preprocess_with_saved_preprocessor(join("data","df_old.csv"), preprocessor)
    X_train_old, X_test_old, y_train_old, y_test_old = split(X_old, y_old)

    mlflow.set_experiment("OPCO_M1_Brief1")
    with mlflow.start_run(run_name="Exp6_new_model_df_old"):
        mlflow.log_param("exp", "6")
        y_pred_test = model_predict(new_model, X_test_old)
        perf_test = evaluate_performance(y_test_old, y_pred_test)
        print_data(perf_test, exp_name="Exp 6 - new model sur df_old")

        mlflow.log_metrics({k: perf_test.get(k) for k in perf_test})

if __name__ == "__main__":
    model_old, preprocessor_old, X_train_old, X_test_old, y_train_old, y_test_old = exp1_baseline_old_model_on_df_old()
    X_train_new, X_test_new, y_train_new, y_test_new = exp2_old_model_on_df_new(model_old, preprocessor_old)
    model_old2 = exp3_retrain_old_model_on_df_old(model_old, X_train_old, y_train_old, X_test_old, y_test_old)
    model_old3 = exp4_retrain_old_model_on_df_new(model_old, X_train_new, y_train_new, X_test_new, y_test_new)
    model_new = exp5_new_model_on_df_new(X_train_new, y_train_new, X_test_new, y_test_new)
    exp6_new_model_on_df_old(model_new, preprocessor_old)
