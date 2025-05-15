import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Impostazioni per una migliore visualizzazione dell'output di pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 0. Definizioni e Costanti ---

# Percorso base per tutti i file di output e input specifici per le predizioni
BASE_OUTPUT_PATH = r'C:\Users\dyshkantiuk_andrii\Desktop\eurobet_virtual_2\PREDICT_RESULT'
# Crea la cartella base se non esiste
if not os.path.exists(BASE_OUTPUT_PATH):
    os.makedirs(BASE_OUTPUT_PATH)
    print(f"Cartella creata: {BASE_OUTPUT_PATH}")

# File per l'addestramento (presumibilmente non in BASE_OUTPUT_PATH, ma nella directory dello script o specificato)
# Assicurati che questo file contenga le colonne 'odds_1', 'odds_X', 'odds_2' se vuoi usarle.
DATA_FILE = 'virtual_matches_data.csv'

# File di input per le nuove predizioni
NEW_PREDICTIONS_INPUT_FILE = os.path.join(BASE_OUTPUT_PATH, 'nuove_partite_da_predire.csv')

# File di output aggiornati per essere salvati nella cartella specificata
NEW_PREDICTIONS_OUTPUT_FILE = os.path.join(BASE_OUTPUT_PATH, 'risultati_predizioni_multiclass_nuove_partite.csv')
MODEL_SAVE_PATH = os.path.join(BASE_OUTPUT_PATH, "catboost_multiclass_predictor.cbm")
VALIDATION_SET_DATA_PATH = os.path.join(BASE_OUTPUT_PATH, 'validation_dataset_multiclass_data.csv')
VALIDATION_RESULTS_EXCEL_PATH = os.path.join(BASE_OUTPUT_PATH, 'validation_evaluation_multiclass_report.xlsx')

TARGET_COLUMN = 'original_result'  # Target per predire 1, X, 2
# REFACTOR: Utilizzo di odds_1, odds_X, e odds_2 come features
FEATURES_TO_USE = ['odds_1', 'odds_X', 'odds_2']
# Assicurati che il tuo DATA_FILE (virtual_matches_data.csv) contenga queste colonne.
# Se non le contiene, lo script darà errore o dovrai adattare FEATURES_TO_USE.

# Mappatura iniziale per le classi target (verrà aggiornata dinamicamente)
TARGET_LABELS = {0: '1', 1: 'X', 2: '2'}  # Esempio, sarà sovrascritto
TARGET_NAMES = list(TARGET_LABELS.values())


# --- 1. Funzioni Utilità ---

def load_data(file_path: str, purpose: str = "training") -> pd.DataFrame | None:
    """Carica i dati da un file CSV."""
    print(f"--- Caricamento Dati da '{file_path}' per {purpose} ---")
    try:
        df = pd.read_csv(file_path)
        print("Prime 5 righe del dataset:")
        print(df.head())
        if purpose == "training":
            print("\nInformazioni sul dataset:")
            df.info()
            print("\nStatistiche descrittive (prima del preprocessing delle quote):")
            # Mostra statistiche solo per le colonne numeriche rilevanti se esistono
            relevant_cols = [col for col in FEATURES_TO_USE + ['home_goals', 'away_goals'] if
                             col in df.columns and pd.api.types.is_numeric_dtype(df[col])]
            if relevant_cols:
                print(df[relevant_cols].describe())
            else:
                print("Nessuna colonna numerica rilevante trovata per le statistiche descrittive.")

        return df
    except FileNotFoundError:
        print(f"Errore: Il file '{file_path}' non è stato trovato. Assicurati che sia nella directory corretta.")
        return None
    except Exception as e:
        print(f"Errore durante il caricamento del file '{file_path}': {e}")
        return None


def preprocess_data_and_create_target(df: pd.DataFrame, target_col_name: str, features_list_param: list) -> tuple[
                                                                                                                pd.DataFrame, pd.Series, pd.DataFrame, list, LabelEncoder] | \
                                                                                                            tuple[
                                                                                                                None, None, None, None, None]:
    """Esegue il preprocessing, crea la colonna target, seleziona le feature e restituisce le feature categoriche."""
    print("\n--- Preprocessing Dati e Creazione Target (per il training) ---")
    df_processed = df.copy()
    # Lavora con una copia della lista di features per evitare modifiche all'originale passata per riferimento
    current_features_list = list(features_list_param)

    # Creazione 'original_result' se non esiste già
    if 'home_goals' in df_processed.columns and 'away_goals' in df_processed.columns:
        if target_col_name not in df_processed.columns:
            def determine_original_result(row):
                if pd.isna(row['home_goals']) or pd.isna(row['away_goals']): return np.nan
                if row['home_goals'] > row['away_goals']:
                    return '1'
                elif row['home_goals'] < row['away_goals']:
                    return '2'
                else:
                    return 'X'

            df_processed[target_col_name] = df_processed.apply(determine_original_result, axis=1)
            print(f"Colonna '{target_col_name}' creata da home_goals e away_goals.")
    elif target_col_name not in df_processed.columns:
        print(f"Errore: Colonne 'home_goals'/'away_goals' mancanti e '{target_col_name}' non presente nel CSV.")
        return None, None, None, None, None

    df_processed.dropna(subset=[target_col_name], inplace=True)
    print(f"Valori unici in '{target_col_name}' prima della codifica: {df_processed[target_col_name].unique()}")

    # REFACTOR: Normalizzazione delle quote se sono intere (es. 250 -> 2.50)
    temp_actual_features_present = []
    for feature in current_features_list:
        if feature in df_processed.columns:
            if feature.startswith('odds_') and df_processed[feature].dtype in ['int64', 'int32']:
                if df_processed[feature].abs().max() > 20:
                    print(f"Conversione della feature '{feature}' da int a float (divisione per 100.0)")
                    df_processed[feature] = df_processed[feature] / 100.0
                else:
                    print(
                        f"La feature '{feature}' è intera ma i valori sembrano già in scala float (es. 2, 3). Convertita a float.")
                    df_processed[feature] = df_processed[feature].astype(float)
                temp_actual_features_present.append(feature)
            elif pd.api.types.is_numeric_dtype(df_processed[feature].dtype):  # Se è già float o altro numerico
                temp_actual_features_present.append(feature)
            elif df_processed[feature].dtype == 'object':
                try:
                    df_processed[feature] = pd.to_numeric(df_processed[feature], errors='raise')
                    print(f"Convertita feature '{feature}' da object a numeric.")
                    temp_actual_features_present.append(feature)
                except ValueError:
                    print(
                        f"ATTENZIONE: La feature '{feature}' è di tipo object e non può essere convertita in numerica. Sarà esclusa.")
            else:  # Tipo non gestito, la escludiamo
                print(
                    f"ATTENZIONE: La feature '{feature}' ha un tipo non gestito ({df_processed[feature].dtype}) e sarà esclusa.")
        else:
            print(f"ATTENZIONE: La feature richiesta '{feature}' non è presente nel DataFrame caricato.")

    actual_features_present = temp_actual_features_present  # Aggiorna la lista delle feature effettivamente utilizzabili

    if not actual_features_present:  # Se nessuna feature è rimasta
        print("Errore: Nessuna feature valida trovata per l'addestramento dopo il preprocessing.")
        return None, None, None, None, None

    print(f"Statistiche descrittive per le features selezionate (dopo normalizzazione/conversione):")
    print(df_processed[actual_features_present].describe())

    le = LabelEncoder()
    try:
        df_processed[target_col_name + '_encoded'] = le.fit_transform(df_processed[target_col_name])
        global TARGET_LABELS, TARGET_NAMES  # Per aggiornare le variabili globali
        TARGET_LABELS = {i: str(label) for i, label in enumerate(le.classes_)}
        TARGET_NAMES = [TARGET_LABELS[i] for i in range(len(TARGET_LABELS))]
        print("TARGET_LABELS aggiornato dinamicamente in base ai dati:")
        print(TARGET_LABELS)
    except Exception as e:
        print(f"Errore durante la codifica del target: {e}")
        return None, None, None, None, None

    print(f"Mappatura della colonna target '{target_col_name}' (effettiva):")
    for original_label, encoded_label in zip(le.classes_, le.transform(le.classes_)):
        print(f"  '{original_label}' -> {encoded_label} (Etichetta: {TARGET_LABELS.get(encoded_label, 'Sconosciuta')})")

    print(
        f"Distribuzione target '{target_col_name}' (codificato):\n{df_processed[target_col_name + '_encoded'].value_counts(normalize=True)}")

    X = df_processed[actual_features_present]
    y = df_processed[target_col_name + '_encoded']
    print(f"Feature effettivamente selezionate per il modello: {actual_features_present}")

    categorical_features = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
    print(f"Feature categoriche identificate: {categorical_features if categorical_features else 'Nessuna'}")

    return X, y, df_processed, categorical_features, le


def run_stratified_kfold_cv(X_train_val_cv: pd.DataFrame, y_train_val_cv: pd.Series, cat_features: list,
                            n_splits: int = 5) -> None:
    """Esegue la Cross-Validation Stratificata K-Fold per multi-classe."""
    print("\n--- Inizio Cross-Validation Stratificata (K-Fold) ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    fold_metrics = {'accuracy': [], 'auc_macro_ovr': [], 'f1_macro': []}

    model_params_cv = dict(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='MultiClass', eval_metric='Accuracy',
        random_seed=42, logging_level='Silent', early_stopping_rounds=50,
    )
    if cat_features: model_params_cv['cat_features'] = cat_features

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val_cv, y_train_val_cv)):
        print(f"Fold {fold + 1}/{n_splits}")
        X_cv_train, X_cv_val = X_train_val_cv.iloc[train_idx], X_train_val_cv.iloc[val_idx]
        y_cv_train, y_cv_val = y_train_val_cv.iloc[train_idx], y_train_val_cv.iloc[val_idx]

        model_fold = CatBoostClassifier(**model_params_cv)
        model_fold.fit(X_cv_train, y_cv_train, eval_set=(X_cv_val, y_cv_val), verbose=0)

        preds_fold = model_fold.predict(X_cv_val).flatten()
        preds_proba_fold = model_fold.predict_proba(X_cv_val)

        unique_labels_fold = np.unique(y_cv_val)

        fold_metrics['accuracy'].append(accuracy_score(y_cv_val, preds_fold))
        try:
            fold_metrics['auc_macro_ovr'].append(
                roc_auc_score(y_cv_val, preds_proba_fold, multi_class='ovr', average='macro',
                              labels=unique_labels_fold))
        except ValueError as e:
            print(f"Attenzione: Impossibile calcolare AUC per il fold {fold + 1}: {e}")
            fold_metrics['auc_macro_ovr'].append(np.nan)

        fold_metrics['f1_macro'].append(
            f1_score(y_cv_val, preds_fold, average='macro', zero_division=0, labels=unique_labels_fold))
        print(
            f"Acc: {fold_metrics['accuracy'][-1]:.4f}, AUC (Macro OVR): {fold_metrics['auc_macro_ovr'][-1]:.4f}, F1 (Macro): {fold_metrics['f1_macro'][-1]:.4f}")

    print(
        f"\nAccuratezza media K-Fold: {np.nanmean(fold_metrics['accuracy']):.4f} (+/- {np.nanstd(fold_metrics['accuracy']):.4f})")
    print(
        f"AUC (Macro OVR) medio K-Fold: {np.nanmean(fold_metrics['auc_macro_ovr']):.4f} (+/- {np.nanstd(fold_metrics['auc_macro_ovr']):.4f})")
    print(
        f"F1-Score (Macro) medio K-Fold: {np.nanmean(fold_metrics['f1_macro']):.4f} (+/- {np.nanstd(fold_metrics['f1_macro']):.4f})")
    print("--- Fine Cross-Validation ---")


def train_final_catboost_model(X_train: pd.DataFrame, y_train: pd.Series,
                               X_val: pd.DataFrame, y_val: pd.Series,
                               cat_features: list) -> CatBoostClassifier:
    """Addestra il modello CatBoost finale per multi-classe."""
    print("\n--- Addestramento del Modello CatBoost Finale (Multi-Classe) ---")
    final_model_params = dict(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
        loss_function='MultiClass', eval_metric='Accuracy',
        random_seed=42, logging_level='Silent', early_stopping_rounds=100,
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
                          label_encoder: LabelEncoder,
                          save_excel_path: str | None = None,
                          X_original_for_excel: pd.DataFrame | None = None) -> dict:
    """Valuta il modello multi-classe su un dato set e opzionalmente salva i risultati in Excel."""
    print(f"\n--- Valutazione Modello su {set_name} Set (Multi-Classe) ---")
    y_pred = model.predict(X_set).flatten()
    y_pred_proba = model.predict_proba(X_set)

    unique_encoded_labels_in_y_set = sorted(y_set.unique())
    current_report_target_names = [TARGET_LABELS.get(encoded_label, f"Classe Sconosciuta {encoded_label}")
                                   for encoded_label in unique_encoded_labels_in_y_set]

    metrics = {
        'accuracy': accuracy_score(y_set, y_pred),
        'auc_macro_ovr': roc_auc_score(y_set, y_pred_proba, multi_class='ovr', average='macro',
                                       labels=unique_encoded_labels_in_y_set),
        'f1_macro': f1_score(y_set, y_pred, average='macro', zero_division=0, labels=unique_encoded_labels_in_y_set),
        'classification_report': classification_report(y_set, y_pred, target_names=current_report_target_names,
                                                       zero_division=0, output_dict=True,
                                                       labels=unique_encoded_labels_in_y_set),
        'confusion_matrix': confusion_matrix(y_set, y_pred, labels=unique_encoded_labels_in_y_set)
    }

    print(f"Accuratezza su {set_name} Set: {metrics['accuracy']:.4f}")
    print(f"AUC (Macro OVR) su {set_name} Set: {metrics['auc_macro_ovr']:.4f}")
    print(f"F1-Score (Macro) su {set_name} Set: {metrics['f1_macro']:.4f}")
    print(
        f"\nClassification Report su {set_name} Set:\n{classification_report(y_set, y_pred, target_names=current_report_target_names, zero_division=0, labels=unique_encoded_labels_in_y_set)}")

    print(f"\nMatrice di Confusione su {set_name} Set:")
    print(metrics['confusion_matrix'])
    plt.figure(figsize=(max(8, len(current_report_target_names) * 2), max(6, len(current_report_target_names) * 1.5)))
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d',
                cmap='Greens' if 'Validazione' in set_name else 'Blues',
                xticklabels=current_report_target_names, yticklabels=current_report_target_names)
    plt.xlabel('Etichetta Predetta');
    plt.ylabel('Etichetta Reale')
    plt.title(f'Matrice di Confusione su {set_name} Set (Target: {TARGET_COLUMN})')
    cm_plot_path = os.path.join(BASE_OUTPUT_PATH, f'confusion_matrix_{set_name.lower().replace(" ", "_")}.png')
    try:
        plt.savefig(cm_plot_path)
        print(f"Grafico matrice di confusione salvato in: {cm_plot_path}")
    except Exception as e:
        print(f"Errore nel salvataggio del grafico della matrice di confusione: {e}")
    plt.show()

    if save_excel_path and X_original_for_excel is not None:
        print(f"\n--- Salvataggio Risultati di {set_name} su Excel ---")
        results_df = X_original_for_excel.copy()

        if TARGET_COLUMN + '_encoded' in results_df.columns:
            results_df = results_df.drop(columns=[TARGET_COLUMN + '_encoded'])

        results_df['actual_result_encoded'] = y_set.reset_index(drop=True)
        results_df['actual_result_original'] = label_encoder.inverse_transform(y_set.reset_index(drop=True))

        results_df['predicted_result_encoded'] = y_pred
        results_df['predicted_result_original'] = label_encoder.inverse_transform(y_pred)

        for i, class_original_label in enumerate(label_encoder.classes_):
            results_df[f'proba_{class_original_label}'] = y_pred_proba[:, i]

        summary_df = pd.DataFrame({
            'Metric': ['Accuracy', 'AUC (Macro OVR)', 'F1 Score (Macro)'],
            'Value': [f"{metrics['accuracy']:.4f}", f"{metrics['auc_macro_ovr']:.4f}", f"{metrics['f1_macro']:.4f}"]
        })
        report_df = pd.DataFrame(metrics['classification_report']).transpose()

        cm_df = pd.DataFrame(metrics['confusion_matrix'],
                             index=[f"Actual {l}" for l in current_report_target_names],
                             columns=[f"Predicted {l}" for l in current_report_target_names])
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
    """Mostra l'importanza delle feature del modello e salva il grafico."""
    print("\n--- Analisi delle Feature Importance (Modello Finale Multi-Classe) ---")
    if hasattr(model, 'get_feature_importance') and feature_names:
        try:
            importances = model.get_feature_importance()
            if len(importances) == len(feature_names):
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importances
                }).sort_values(by='importance', ascending=False)
                print(importance_df)

                if not importance_df.empty:
                    plt.figure(figsize=(10, max(4, len(feature_names) * 0.5)))
                    sns.barplot(x='importance', y='feature', data=importance_df, palette="viridis")
                    plt.title(f'Importanza delle Feature (Predizione {TARGET_COLUMN})')
                    plt.tight_layout()
                    fi_plot_path = os.path.join(BASE_OUTPUT_PATH, 'feature_importance.png')
                    try:
                        plt.savefig(fi_plot_path)
                        print(f"Grafico importanza feature salvato in: {fi_plot_path}")
                    except Exception as e:
                        print(f"Errore nel salvataggio del grafico importanza feature: {e}")
                    plt.show()
                else:
                    print("DataFrame dell'importanza delle feature è vuoto.")
            else:
                print(
                    f"Numero di feature names ({len(feature_names)}) non corrisponde al numero di importances ({len(importances)}).")

        except Exception as e:
            print(f"Errore durante il calcolo o la visualizzazione dell'importanza delle feature: {e}")
    else:
        print("Nessuna feature specificata o metodo get_feature_importance non disponibile/applicabile.")


def predict_results_on_new_data(new_data_df: pd.DataFrame, model_path: str, features_list_param: list,
                                label_encoder: LabelEncoder) -> pd.DataFrame | None:
    """Carica un modello salvato e fa predizioni multi-classe su nuovi dati."""
    print(f"\n--- Predizione Multi-Classe su Nuovi Dati (da '{model_path}') ---")

    X_new_processed = new_data_df.copy()
    current_features_list = list(features_list_param)  # Lavora con una copia

    temp_actual_features_present_new_data = []
    for feature in current_features_list:
        if feature in X_new_processed.columns:
            if feature.startswith('odds_') and X_new_processed[feature].dtype in ['int64', 'int32']:
                if X_new_processed[feature].abs().max() > 20:
                    print(f"Conversione della feature '{feature}' nei nuovi dati (divisione per 100.0)")
                    X_new_processed[feature] = X_new_processed[feature] / 100.0
                else:
                    X_new_processed[feature] = X_new_processed[feature].astype(float)
                temp_actual_features_present_new_data.append(feature)
            elif pd.api.types.is_numeric_dtype(X_new_processed[feature].dtype):
                temp_actual_features_present_new_data.append(feature)
            elif X_new_processed[feature].dtype == 'object':
                try:
                    X_new_processed[feature] = pd.to_numeric(X_new_processed[feature], errors='raise')
                    temp_actual_features_present_new_data.append(feature)
                except ValueError:
                    print(
                        f"ATTENZIONE: La feature '{feature}' nei nuovi dati è object e non convertibile. Potrebbe causare errori se usata dal modello.")
            else:
                print(
                    f"ATTENZIONE: La feature '{feature}' nei nuovi dati ha un tipo non gestito ({X_new_processed[feature].dtype}).")
        else:
            print(f"Errore: La feature richiesta '{feature}' non è presente nei nuovi dati.")
            return None

    actual_features_present_new_data = temp_actual_features_present_new_data

    # Verifica che tutte le features attese dal modello siano presenti e processate
    if not all(feature in actual_features_present_new_data for feature in features_list_param):
        missing_cols = [f for f in features_list_param if f not in actual_features_present_new_data]
        print(
            f"Errore: I nuovi dati (dopo il preprocessing) non contengono tutte le features attese dal modello: {features_list_param}. Mancanti: {missing_cols}")
        return None

    X_new = X_new_processed[actual_features_present_new_data]  # Usa solo le features processate e attese

    loaded_model = load_catboost_model(model_path)
    if loaded_model is None:
        return None

    print("Effettuazione predizioni sui nuovi dati...")
    try:
        predictions_encoded = loaded_model.predict(X_new).flatten()
        probabilities = loaded_model.predict_proba(X_new)

        results_df = new_data_df.copy()
        results_df['predicted_result_encoded'] = predictions_encoded
        results_df['predicted_result'] = label_encoder.inverse_transform(predictions_encoded)

        for i, class_original_label in enumerate(label_encoder.classes_):
            results_df[f'probability_{class_original_label}'] = probabilities[:, i]

        print("Predizioni completate.")
        return results_df
    except Exception as e:
        print(f"Errore durante la predizione su nuovi dati: {e}")
        return None


def create_sample_prediction_file_if_not_exists(file_path: str, features_to_use_in_model: list):
    """Crea un file CSV di esempio per le predizioni se non esiste già, includendo le features del modello."""
    if not os.path.exists(file_path):
        print(f"Il file '{file_path}' non esiste. Creazione di un file di esempio...")
        sample_data = {'match_id': [101, 102, 103, 104, 105, 106, 107, 108]}

        if 'odds_1' in features_to_use_in_model:
            sample_data['odds_1'] = [1.50, 2.00, 2.50, 3.00, 3.50, 4.00, 2.20, 2.80]
        if 'odds_X' in features_to_use_in_model:
            sample_data['odds_X'] = [3.20, 3.00, 3.50, 3.80, 4.00, 4.20, 3.10, 3.30]
        if 'odds_2' in features_to_use_in_model:
            sample_data['odds_2'] = [5.00, 4.00, 3.00, 2.50, 2.20, 2.00, 3.50, 2.70]

        for feature in features_to_use_in_model:
            if feature not in sample_data:  # Se una feature diversa da odds_1,X,2 è usata
                sample_data[feature] = [round(np.random.uniform(1.0, 5.0), 2) for _ in range(8)]

        sample_df = pd.DataFrame(sample_data)
        try:
            sample_df.to_csv(file_path, index=False)
            print(f"File di esempio '{file_path}' creato con successo con colonne: {list(sample_df.columns)}.")
        except Exception as e:
            print(f"Errore durante la creazione del file di esempio '{file_path}': {e}")
    else:
        print(f"Il file '{file_path}' esiste già. Verrà utilizzato per le predizioni.")


# --- 2. Script Principale ---
if __name__ == "__main__":
    # STEP 1: Caricamento Dati per Training
    df_original = load_data(DATA_FILE, purpose="training")
    if df_original is None:
        if not os.path.exists(DATA_FILE):
            print(f"Creazione di un file '{DATA_FILE}' di esempio nella directory dello script...")
            sample_training_data = {
                'home_team': [f'TeamA{i}' for i in range(200)],
                'away_team': [f'TeamB{i}' for i in range(200)],
                'home_goals': np.random.choice([0, 1, 2, 3, 1, 0, 2, 1, 1, 0, 0, 1, 2, 2, 3, 0], 200),
                'away_goals': np.random.choice([0, 1, 2, 0, 1, 1, 0, 2, 0, 1, 2, 2, 1, 0, 0, 3], 200),
                'odds_1': np.random.randint(120, 500, 200),
                'odds_X': np.random.randint(280, 600, 200),
                'odds_2': np.random.randint(150, 700, 200)
            }
            sample_training_data['score'] = [f"{hg}-{ag}" for hg, ag in zip(sample_training_data['home_goals'],
                                                                            sample_training_data['away_goals'])]
            sample_training_data['date'] = pd.to_datetime('today').strftime('%d-%m-%Y')

            sample_training_df = pd.DataFrame(sample_training_data)
            try:
                sample_training_df.to_csv(DATA_FILE, index=False)
                print(
                    f"'{DATA_FILE}' di esempio creato con colonne: {list(sample_training_df.columns)}. Riesegui lo script.")
            except Exception as e:
                print(f"Errore durante la creazione del file di training di esempio '{DATA_FILE}': {e}")
        exit()

    # STEP 2: Preprocessing e Creazione Target per Training
    X, y, df_processed, categorical_features_for_model, label_enc = preprocess_data_and_create_target(
        df_original, TARGET_COLUMN, FEATURES_TO_USE
    )
    if X is None or y is None or label_enc is None:
        print("Preprocessing o creazione target falliti. Uscita.")
        exit()

    CURRENT_FEATURES_USED = list(X.columns)  # Features effettivamente usate dopo il preprocessing

    # STEP 3: Divisione Dati per Training/Validazione/Test
    print("\n--- Divisione dei Dati (per Training/Validazione/Test) ---")

    # Colonne da includere nello split: quelle usate come features + target originale + target codificato
    # Assicurati che TARGET_COLUMN e TARGET_COLUMN+'_encoded' siano in df_processed
    cols_for_split = CURRENT_FEATURES_USED + [col for col in [TARGET_COLUMN, TARGET_COLUMN + '_encoded'] if
                                              col in df_processed.columns]
    cols_for_split = sorted(list(set(cols_for_split)))

    missing_cols_in_df = [col for col in cols_for_split if col not in df_processed.columns]
    if missing_cols_in_df:
        print(f"Errore: Colonne necessarie per lo split ({missing_cols_in_df}) non trovate in df_processed.")
        exit()

    try:
        df_train_val, df_test = train_test_split(
            df_processed[cols_for_split], test_size=0.20, random_state=42,
            stratify=df_processed[TARGET_COLUMN + '_encoded']
        )
        df_train_final, df_validation_final = train_test_split(
            df_train_val, test_size=0.25, random_state=42, stratify=df_train_val[TARGET_COLUMN + '_encoded']
        )
    except ValueError as e:
        print(f"Errore durante la divisione dei dati (train_test_split): {e}")
        print(
            f"Distribuzione classi nel set da dividere (df_processed):\n{df_processed[TARGET_COLUMN + '_encoded'].value_counts()}")
        if TARGET_COLUMN + '_encoded' in df_train_val.columns:
            print(
                f"Distribuzione classi nel set da dividere (df_train_val):\n{df_train_val[TARGET_COLUMN + '_encoded'].value_counts()}")
        exit()

    X_train_final = df_train_final[CURRENT_FEATURES_USED]
    y_train_final = df_train_final[TARGET_COLUMN + '_encoded']
    X_validation_final = df_validation_final[CURRENT_FEATURES_USED]
    y_validation_final = df_validation_final[TARGET_COLUMN + '_encoded']
    X_test = df_test[CURRENT_FEATURES_USED]
    y_test = df_test[TARGET_COLUMN + '_encoded']

    print(f"Dimensioni Training Set: X {X_train_final.shape}, y {y_train_final.shape}")
    print(f"Dimensioni Validation Set: X {X_validation_final.shape}, y {y_validation_final.shape}")
    print(f"Dimensioni Test Set: X {X_test.shape}, y {y_test.shape}")

    validation_set_to_save = df_validation_final.copy()
    try:
        validation_set_to_save.to_csv(VALIDATION_SET_DATA_PATH, index=False)
        print(f"Set di validazione (dati) salvato in: {VALIDATION_SET_DATA_PATH}")
    except Exception as e:
        print(f"Errore salvataggio set di validazione: {e}")

    # STEP 4: K-Fold Cross-Validation
    # CORREZIONE: Usare df_train_val invece di X_train_val
    if not df_train_val[CURRENT_FEATURES_USED].empty:
        run_stratified_kfold_cv(df_train_val[CURRENT_FEATURES_USED], df_train_val[TARGET_COLUMN + '_encoded'],
                                categorical_features_for_model)
    else:
        print("Skipping K-Fold Cross-Validation due a df_train_val o features vuote.")

    # STEP 5: Addestramento Modello Finale
    if X_train_final.empty or y_train_final.empty:
        print("Dati di training finali vuoti. Impossibile addestrare il modello.")
        exit()

    final_model = train_final_catboost_model(
        X_train_final, y_train_final,
        X_validation_final, y_validation_final,
        categorical_features_for_model
    )

    # STEP 6: Salvataggio Modello Finale
    save_catboost_model(final_model, MODEL_SAVE_PATH)

    # STEP 7: Valutazione sul Set di Validazione
    _ = evaluate_model_on_set(final_model, X_validation_final, y_validation_final, "Validazione",
                              label_encoder=label_enc,
                              save_excel_path=VALIDATION_RESULTS_EXCEL_PATH,
                              X_original_for_excel=df_validation_final)

    # STEP 8: Valutazione sul Set di Test
    test_metrics = evaluate_model_on_set(final_model, X_test, y_test, "Test",
                                         label_encoder=label_enc,
                                         X_original_for_excel=df_test)
    print("\n--- Dettaglio Risultati sul Test Set (Multi-Classe) ---")
    print(f"Numero totale di partite nel Test Set: {len(y_test)}")

    # STEP 9: Analisi Feature Importance
    display_feature_importance(final_model, CURRENT_FEATURES_USED)

    # --- STEP 10: Predizione su Nuovi Dati da File ---
    print(f"\n\n--- STEP 10: Predizione su Nuovi Dati da File '{NEW_PREDICTIONS_INPUT_FILE}' ---")
    create_sample_prediction_file_if_not_exists(NEW_PREDICTIONS_INPUT_FILE, CURRENT_FEATURES_USED)
    new_matches_to_predict_df_raw = load_data(NEW_PREDICTIONS_INPUT_FILE, purpose="prediction")

    if new_matches_to_predict_df_raw is not None and not new_matches_to_predict_df_raw.empty:
        if os.path.exists(MODEL_SAVE_PATH):
            predicted_results_df = predict_results_on_new_data(
                new_matches_to_predict_df_raw,
                MODEL_SAVE_PATH,
                CURRENT_FEATURES_USED,
                label_enc
            )
            if predicted_results_df is not None:
                print('\nRisultati delle predizioni sui nuovi dati (prime 5 righe):')
                print(predicted_results_df.head())
                try:
                    predicted_results_df.to_csv(NEW_PREDICTIONS_OUTPUT_FILE, index=False)
                    print(f"Predizioni sui nuovi dati salvate in: '{NEW_PREDICTIONS_OUTPUT_FILE}'")
                except Exception as e:
                    print(f"Errore durante il salvataggio dei risultati delle predizioni: {e}")
            else:
                print("Predizione sui nuovi dati non riuscita.")
        else:
            print(f"Modello '{MODEL_SAVE_PATH}' non trovato. Impossibile effettuare predizioni su nuovi dati.")
    elif new_matches_to_predict_df_raw is None:
        print(f"Impossibile caricare i dati da '{NEW_PREDICTIONS_INPUT_FILE}' per la predizione.")
    else:
        print(f"Il file '{NEW_PREDICTIONS_INPUT_FILE}' è vuoto. Nessun dato su cui predire.")

    # STEP 11: Istruzioni Finali
    print("\n\n--- STEP 11: Istruzioni Finali ---")
    print(f"Il modello multi-classe è stato addestrato usando le features: {CURRENT_FEATURES_USED}")
    print(f"Modello salvato in: {MODEL_SAVE_PATH}")
    print(f"I file di output sono stati salvati in: {BASE_OUTPUT_PATH}")
    print(f"Lo script ha tentato di effettuare predizioni sul file '{NEW_PREDICTIONS_INPUT_FILE}'.")
    print(f"Puoi modificare '{NEW_PREDICTIONS_INPUT_FILE}' con le tue quote (formato float, es. 2.50) e rieseguire.")

    print("\n--- Fine del processo ---")

