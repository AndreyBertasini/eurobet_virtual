import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Impostazioni per una migliore visualizzazione dell'output di pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 0. Definizioni e Costanti ---
DATA_FILE = 'virtual_matches_data.csv'  # File per l'addestramento
NEW_PREDICTIONS_INPUT_FILE = 'nuove_partite_da_predire.csv'  # File per le nuove predizioni
NEW_PREDICTIONS_OUTPUT_FILE = 'risultati_predizioni_nuove_partite.csv'  # File opzionale per salvare i risultati delle nuove predizioni

MODEL_SAVE_PATH = "catboost_draw_predictor.cbm"
VALIDATION_SET_DATA_PATH = 'validation_dataset_data.csv'
VALIDATION_RESULTS_EXCEL_PATH = 'validation_evaluation_report.xlsx'
TARGET_COLUMN = 'is_draw'
FEATURES_TO_USE = ['odds_1']  # Come da richiesta, usare solo odds_1


# --- 1. Funzioni Utilità ---

def load_data(file_path: str, purpose: str = "training") -> pd.DataFrame | None:
    """Carica i dati da un file CSV."""
    print(f"--- Caricamento Dati da '{file_path}' per {purpose} ---")
    try:
        df = pd.read_csv(file_path)
        print("Prime 5 righe del dataset:")
        print(df.head())
        if purpose == "training":  # Mostra info solo per il dataset di training
            print("\nInformazioni sul dataset:")
            df.info()
        return df
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato. Assicurati che sia nella directory corretta.")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento del file '{file_path}': {e}")
        return None


def preprocess_data_and_create_target(df: pd.DataFrame, target_col_name: str, features_list: list) -> tuple[
                                                                                                          pd.DataFrame, pd.Series, pd.DataFrame, list] | \
                                                                                                      tuple[
                                                                                                          None, None, None, None]:
    """Esegue il preprocessing, crea la colonna target, seleziona le feature e restituisce le feature categoriche."""
    print("\n--- Preprocessing Dati e Creazione Target (per il training) ---")
    df_processed = df.copy()

    # Creazione 'original_result'
    if 'home_goals' in df_processed.columns and 'away_goals' in df_processed.columns:
        def determine_original_result(row):
            if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
                return np.nan
            if row['home_goals'] > row['away_goals']:
                return '1'
            elif row['home_goals'] < row['away_goals']:
                return '2'
            else:
                return 'X'

        df_processed['original_result'] = df_processed.apply(determine_original_result, axis=1)
        df_processed.dropna(subset=['original_result'], inplace=True)
        print("Colonna 'original_result' creata.")
    else:
        print("Errore: Colonne 'home_goals' o 'away_goals' mancanti per 'original_result'.")
        return None, None, None, None

    # Creazione target 'is_draw'
    df_processed[target_col_name] = df_processed['original_result'].apply(lambda x: 1 if x == 'X' else 0)
    print(f"Colonna target '{target_col_name}' creata.")
    print(f"Distribuzione target '{target_col_name}':\n{df_processed[target_col_name].value_counts(normalize=True)}")

    # Selezione Feature e Target
    if not all(feature in df_processed.columns for feature in features_list):
        print(f"Errore: Feature richieste ({features_list}) non presenti dopo il preprocessing.")
        return None, None, None, None

    X = df_processed[features_list]
    y = df_processed[target_col_name]
    print(f"Feature selezionate per il modello: {features_list}")

    categorical_features = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
    print(f"Feature categoriche identificate: {categorical_features if categorical_features else 'Nessuna'}")

    return X, y, df_processed, categorical_features


def run_stratified_kfold_cv(X_train_val_cv: pd.DataFrame, y_train_val_cv: pd.Series, cat_features: list,
                            n_splits: int = 5) -> None:
    """Esegue la Cross-Validation Stratificata K-Fold."""
    print("\n--- Inizio Cross-Validation Stratificata (K-Fold) ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    fold_metrics = {'accuracy': [], 'auc': [], 'f1_draw': []}

    model_params_cv = dict(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='Logloss', eval_metric='AUC', random_seed=42,
        logging_level='Silent', early_stopping_rounds=50, auto_class_weights='Balanced'
    )
    if cat_features: model_params_cv['cat_features'] = cat_features

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_cv, y_train_val_cv)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_cv_train, X_cv_val = X_train_val_cv.iloc[train_idx], X_train_val_cv.iloc[val_idx]
        y_cv_train, y_cv_val = y_train_val_cv.iloc[train_idx], y_train_val_cv.iloc[val_idx]

        model_fold = CatBoostClassifier(**model_params_cv)
        model_fold.fit(X_cv_train, y_cv_train, eval_set=(X_cv_val, y_cv_val), verbose=0)

        preds_fold = model_fold.predict(X_cv_val)
        preds_proba_fold = model_fold.predict_proba(X_cv_val)[:, 1]

        fold_metrics['accuracy'].append(accuracy_score(y_cv_val, preds_fold))
        fold_metrics['auc'].append(roc_auc_score(y_cv_val, preds_proba_fold))
        fold_metrics['f1_draw'].append(f1_score(y_cv_val, preds_fold, pos_label=1, zero_division=0))
        print(
            f"Acc: {fold_metrics['accuracy'][-1]:.4f}, AUC: {fold_metrics['auc'][-1]:.4f}, F1 Pareggio: {fold_metrics['f1_draw'][-1]:.4f}")

    print(
        f"\nAccuratezza media K-Fold: {np.mean(fold_metrics['accuracy']):.4f} (+/- {np.std(fold_metrics['accuracy']):.4f})")
    print(f"AUC medio K-Fold: {np.mean(fold_metrics['auc']):.4f} (+/- {np.std(fold_metrics['auc']):.4f})")
    print(
        f"F1-Score Pareggio medio K-Fold: {np.mean(fold_metrics['f1_draw']):.4f} (+/- {np.std(fold_metrics['f1_draw']):.4f})")
    print("--- Fine Cross-Validation ---")


def train_final_catboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               cat_features: list) -> CatBoostClassifier:
    """Addestra il modello CatBoost finale."""
    print("\n--- Addestramento del Modello CatBoost Finale ---")
    final_model_params = dict(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
        loss_function='Logloss', eval_metric='AUC', random_seed=42,
        logging_level='Silent', early_stopping_rounds=100, auto_class_weights='Balanced'
    )
    if cat_features: final_model_params['cat_features'] = cat_features

    model = CatBoostClassifier(**final_model_params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val), verbose=500)
    print("Addestramento modello finale completato.")
    return model


def save_catboost_model(model: CatBoostClassifier, path: str) -> None:
    """Salva il modello CatBoost su disco."""
    print(f"\n--- Salvataggio Modello in '{path}' ---")
    try:
        model.save_model(path)
        print("Modello salvato con successo.")
    except Exception as e:
        print(f"Errore durante il salvataggio del modello: {e}")


def load_catboost_model(path: str) -> CatBoostClassifier | None:
    """Carica un modello CatBoost da disco."""
    print(f"\n--- Caricamento Modello da '{path}' ---")
    try:
        model = CatBoostClassifier()
        model.load_model(path)
        print("Modello caricato con successo.")
        return model
    except Exception as e:
        print(f"Errore durante il caricamento del modello: {e}")
        return None


def evaluate_model_on_set(model: CatBoostClassifier, X_set: pd.DataFrame, y_set: pd.Series, set_name: str,
                          save_excel_path: str | None = None,
                          X_original_for_excel: pd.DataFrame | None = None) -> dict:
    """Valuta il modello su un dato set e opzionalmente salva i risultati in Excel."""
    print(f"\n--- Valutazione Modello su {set_name} Set ---")
    y_pred = model.predict(X_set)
    y_pred_proba = model.predict_proba(X_set)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_set, y_pred),
        'auc': roc_auc_score(y_set, y_pred_proba),
        'f1_draw': f1_score(y_set, y_pred, pos_label=1, zero_division=0),
        'classification_report': classification_report(y_set, y_pred, target_names=['Non Pareggio (0)', 'Pareggio (1)'],
                                                       zero_division=0, output_dict=True),
        'confusion_matrix': confusion_matrix(y_set, y_pred)
    }

    print(f"Accuratezza su {set_name} Set: {metrics['accuracy']:.4f}")
    print(f"AUC su {set_name} Set: {metrics['auc']:.4f}")
    print(f"F1-Score Pareggio su {set_name} Set: {metrics['f1_draw']:.4f}")
    print(
        f"\nClassification Report su {set_name} Set:\n{classification_report(y_set, y_pred, target_names=['Non Pareggio (0)', 'Pareggio (1)'], zero_division=0)}")

    print(f"\nMatrice di Confusione su {set_name} Set:")
    print(metrics['confusion_matrix'])
    plt.figure(figsize=(8, 6))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Greens' if 'Validazione' in set_name else 'Blues',
                xticklabels=['Non Pareggio', 'Pareggio'], yticklabels=['Non Pareggio', 'Pareggio'])
    plt.xlabel('Etichetta Predetta');
    plt.ylabel('Etichetta Reale')
    plt.title(f'Matrice di Confusione su {set_name} Set (Target: is_draw)')
    plt.show()

    if save_excel_path and X_original_for_excel is not None:
        print(f"\n--- Salvataggio Risultati di {set_name} su Excel ---")
        results_df = X_original_for_excel.copy()
        results_df['actual_is_draw'] = y_set.reset_index(drop=True)
        results_df['predicted_is_draw'] = y_pred
        results_df['predicted_proba_draw'] = y_pred_proba

        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC', 'F1 Score (Draw)'],
            'Value': [f"{metrics['accuracy']:.4f}", f"{metrics['auc']:.4f}", f"{metrics['f1_draw']:.4f}"]
        })
        report_df = pd.DataFrame(metrics['classification_report']).transpose()
        cm_df = pd.DataFrame(metrics['confusion_matrix'],
                             index=[f"Actual {l}" for l in ['Non Pareggio (0)', 'Pareggio (1)']],
                             columns=[f"Predicted {l}" for l in ['Non Pareggio (0)', 'Pareggio (1)']])
        try:
            with pd.ExcelWriter(save_excel_path, engine='openpyxl') as writer:
                results_df.to_excel(writer, sheet_name=f'Predictions_{set_name}', index=False)
                summary_df.to_excel(writer, sheet_name=f'Metrics_Summary_{set_name}', index=False)
                report_df.to_excel(writer, sheet_name=f'Class_Report_{set_name}')
                cm_df.to_excel(writer, sheet_name=f'Conf_Matrix_{set_name}')
            print(f"Risultati di {set_name} salvati in: {save_excel_path}")
        except Exception as e:
            print(f"Errore salvataggio Excel per {set_name}: {e}. Assicurati 'openpyxl' sia installato.")
    return metrics


def display_feature_importance(model: CatBoostClassifier, feature_names: list) -> None:
    """Mostra l'importanza delle feature del modello."""
    print("\n--- Analisi delle Feature Importance (Modello Finale) ---")
    if hasattr(model, 'get_feature_importance') and len(feature_names) > 0:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.get_feature_importance()
        }).sort_values(by='importance', ascending=False)
        print(importance_df)

        if len(feature_names) == 1:
            print(
                f"Importanza per la feature '{feature_names[0]}': {importance_df['importance'].iloc[0]:.2f}% (CatBoost assegna 100% a una singola feature)")
        else:
            plt.figure(figsize=(10, max(2, len(feature_names) * 0.3)))
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title(f'Importanza delle Feature (Predizione Pareggio con {feature_names})')
            plt.tight_layout()
            plt.show()
    else:
        print("Nessuna feature o metodo get_feature_importance non disponibile.")


def predict_draws_on_new_data(new_data_df: pd.DataFrame, model_path: str, features_list: list) -> tuple[
    np.ndarray | None, np.ndarray | None]:
    """Carica un modello salvato e fa predizioni su nuovi dati."""
    print(f"\n--- Predizione su Nuovi Dati (da '{model_path}') ---")

    if not all(feature in new_data_df.columns for feature in features_list):
        print(f"Errore: I nuovi dati devono contenere le seguenti colonne: {features_list}")
        missing_cols = [f for f in features_list if f not in new_data_df.columns]
        print(f"Colonne mancanti: {missing_cols}")
        return None, None

    X_new = new_data_df[features_list].copy()

    loaded_model = load_catboost_model(model_path)
    if loaded_model is None:
        return None, None

    print("Effettuazione predizioni sui nuovi dati...")
    try:
        predictions = loaded_model.predict(X_new)
        probabilities = loaded_model.predict_proba(X_new)[:, 1]
        print("Predizioni completate.")
        return predictions, probabilities
    except Exception as e:
        print(f"Errore durante la predizione su nuovi dati: {e}")
        return None, None


def create_sample_prediction_file_if_not_exists(file_path: str, features_to_include: list):
    """Crea un file CSV di esempio per le predizioni se non esiste già."""
    if not os.path.exists(file_path):
        print(f"Il file '{file_path}' non esiste. Creazione di un file di esempio...")
        # Crea dati di esempio solo per le colonne specificate in features_to_include
        # In questo caso, solo 'odds_1'
        if features_to_include == ['odds_1']:
            sample_data = {
                'odds_1': [150, 200, 250, 300, 350, 400, 220, 280]
            }
            sample_df = pd.DataFrame(sample_data)
            try:
                sample_df.to_csv(file_path, index=False)
                print(f"File di esempio '{file_path}' creato con successo con la colonna 'odds_1'.")
            except Exception as e:
                print(f"Errore durante la creazione del file di esempio '{file_path}': {e}")
        else:
            print(f"Logica di creazione file di esempio non implementata per features: {features_to_include}")
    else:
        print(f"Il file '{file_path}' esiste già. Verrà utilizzato per le predizioni.")


# --- 2. Script Principale ---
if __name__ == "__main__":
    # STEP 1: Caricamento Dati per Training
    df_original = load_data(DATA_FILE, purpose="training")
    if df_original is None:
        exit()

    # STEP 2: Preprocessing e Creazione Target per Training
    X, y, df_processed, categorical_features_for_model = preprocess_data_and_create_target(
        df_original, TARGET_COLUMN, FEATURES_TO_USE
    )
    if X is None or y is None:
        exit()

    # STEP 3: Divisione Dati per Training/Validazione/Test
    print("\n--- Divisione dei Dati (per Training/Validazione/Test) ---")
    df_train_val, df_test = train_test_split(
        df_processed, test_size=0.20, random_state=42, stratify=df_processed[TARGET_COLUMN]
    )
    df_train_final, df_validation_final = train_test_split(
        df_train_val, test_size=0.25, random_state=42, stratify=df_train_val[TARGET_COLUMN]
    )

    X_train_final = df_train_final[FEATURES_TO_USE]
    y_train_final = df_train_final[TARGET_COLUMN]
    X_validation_final = df_validation_final[FEATURES_TO_USE]
    y_validation_final = df_validation_final[TARGET_COLUMN]
    X_test = df_test[FEATURES_TO_USE]
    y_test = df_test[TARGET_COLUMN]

    print(f"Dimensioni Training Set: X {X_train_final.shape}, y {y_train_final.shape}")
    print(f"Dimensioni Validation Set: X {X_validation_final.shape}, y {y_validation_final.shape}")
    print(f"Dimensioni Test Set: X {X_test.shape}, y {y_test.shape}")

    validation_set_to_save = df_validation_final[FEATURES_TO_USE + [TARGET_COLUMN]].copy()
    try:
        validation_set_to_save.to_csv(VALIDATION_SET_DATA_PATH, index=False)
        print(f"Set di validazione (dati) salvato in: {VALIDATION_SET_DATA_PATH}")
    except Exception as e:
        print(f"Errore salvataggio set di validazione: {e}")

    # STEP 4: K-Fold Cross-Validation
    X_tv_cv = df_train_val[FEATURES_TO_USE]
    y_tv_cv = df_train_val[TARGET_COLUMN]
    run_stratified_kfold_cv(X_tv_cv, y_tv_cv, categorical_features_for_model)

    # STEP 5: Addestramento Modello Finale
    final_model = train_final_catboost_model(
        X_train_final, y_train_final,
        X_validation_final, y_validation_final,
        categorical_features_for_model
    )

    # STEP 6: Salvataggio Modello Finale
    save_catboost_model(final_model, MODEL_SAVE_PATH)

    # STEP 7: Valutazione sul Set di Validazione
    _ = evaluate_model_on_set(final_model, X_validation_final, y_validation_final, "Validazione",
                              save_excel_path=VALIDATION_RESULTS_EXCEL_PATH,
                              X_original_for_excel=X_validation_final)

    # STEP 8: Valutazione sul Set di Test
    test_metrics = evaluate_model_on_set(final_model, X_test, y_test, "Test")
    TN_test = test_metrics['confusion_matrix'][0, 0]
    FP_test = test_metrics['confusion_matrix'][0, 1]
    FN_test = test_metrics['confusion_matrix'][1, 0]
    TP_test = test_metrics['confusion_matrix'][1, 1]
    print("\n--- Dettaglio Predizione Pareggi sul Test Set ---")
    print(f"Numero totale di partite nel Test Set: {len(y_test)}")
    print(f"Numero di pareggi effettivi nel Test Set: {np.sum(y_test)}")
    print(f"Pareggi effettivi previsti correttamente (TP): {TP_test}")
    print(f"Pareggi effettivi previsti erroneamente come 'Non Pareggio' (FN - Pareggi Mancati): {FN_test}")
    print(f"Non-pareggi effettivi previsti erroneamente come 'Pareggio' (FP - Falsi Allarmi): {FP_test}")
    print(f"Non-pareggi effettivi previsti correttamente (TN): {TN_test}")

    # STEP 9: Analisi Feature Importance
    display_feature_importance(final_model, FEATURES_TO_USE)

    # --- STEP 10: Predizione su Nuovi Dati da File ---
    print(f"\n\n--- STEP 10: Predizione su Nuovi Dati da File '{NEW_PREDICTIONS_INPUT_FILE}' ---")

    # Assicura che il file di input per le nuove predizioni esista (crea un esempio se necessario)
    create_sample_prediction_file_if_not_exists(NEW_PREDICTIONS_INPUT_FILE, FEATURES_TO_USE)

    # Carica i nuovi dati per la predizione
    new_matches_to_predict_df = load_data(NEW_PREDICTIONS_INPUT_FILE, purpose="prediction")

    if new_matches_to_predict_df is not None and not new_matches_to_predict_df.empty:
        # Verifica che il modello sia stato salvato prima di tentare di caricarlo per la predizione
        if os.path.exists(MODEL_SAVE_PATH):
            predictions, probabilities = predict_draws_on_new_data(
                new_matches_to_predict_df,
                MODEL_SAVE_PATH,
                FEATURES_TO_USE
            )
            if predictions is not None and probabilities is not None:
                new_matches_to_predict_df['predicted_is_draw'] = predictions
                new_matches_to_predict_df['probability_of_draw'] = probabilities
                print('\nRisultati delle predizioni sui nuovi dati (prime 5 righe):')
                print(new_matches_to_predict_df.head())

                # Opzionale: Salva i risultati delle predizioni in un nuovo file CSV
                try:
                    new_matches_to_predict_df.to_csv(NEW_PREDICTIONS_OUTPUT_FILE, index=False)
                    print(f"Predizioni sui nuovi dati salvate in: '{NEW_PREDICTIONS_OUTPUT_FILE}'")
                except Exception as e:
                    print(f"Errore durante il salvataggio dei risultati delle predizioni: {e}")
            else:
                print("Predizione sui nuovi dati non riuscita.")
        else:
            print(f"Modello '{MODEL_SAVE_PATH}' non trovato. Impossibile effettuare predizioni su nuovi dati.")
    elif new_matches_to_predict_df is None:
        print(f"Impossibile caricare i dati da '{NEW_PREDICTIONS_INPUT_FILE}' per la predizione.")
    else:  # DataFrame vuoto
        print(f"Il file '{NEW_PREDICTIONS_INPUT_FILE}' è vuoto. Nessun dato su cui predire.")

    # STEP 11: Istruzioni Finali
    print("\n\n--- STEP 11: Istruzioni Finali ---")
    print(f"Il modello è stato addestrato e salvato in: {MODEL_SAVE_PATH}")
    print(f"Lo script ha anche tentato di effettuare predizioni sul file '{NEW_PREDICTIONS_INPUT_FILE}'.")
    print("Se il file non esisteva, ne è stato creato uno di esempio.")
    print(
        f"I risultati di queste predizioni (se riuscite) sono stati stampati nel log e opzionalmente salvati in '{NEW_PREDICTIONS_OUTPUT_FILE}'.")
    print("Puoi modificare il file '{NEW_PREDICTIONS_INPUT_FILE}' con le tue quote 'odds_1' e rieseguire lo script.")
    print(
        "Per utilizzare il modello in un altro contesto, carica il file .cbm e usa la funzione 'predict_draws_on_new_data'.")

    print("\n--- Fine del processo ---")

