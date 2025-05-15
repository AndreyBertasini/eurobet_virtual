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
DATA_FILE = 'virtual_matches_data.csv'  # File per l'addestramento del modello di produzione
NEW_PREDICTIONS_INPUT_FILE = 'nuove_partite_da_predire.csv'  # File per le nuove predizioni
NEW_PREDICTIONS_OUTPUT_FILE = 'risultati_predizioni_nuove_partite.csv'  # File per salvare i risultati delle nuove predizioni

MODEL_SAVE_PATH = "catboost_draw_predictor_production.cbm"  # Nome modificato per chiarezza
TARGET_COLUMN = 'is_draw'
FEATURES_TO_USE = ['odds_1']  # Feature usata per il modello


# --- 1. Funzioni Utilità ---

def load_data(file_path: str, purpose: str = "generic") -> pd.DataFrame | None:
    """Carica i dati da un file CSV."""
    print(f"--- Caricamento Dati da '{file_path}' (scopo: {purpose}) ---")
    try:
        df = pd.read_csv(file_path)
        print(f"Prime 5 righe del dataset '{file_path}':")
        print(df.head())
        if purpose == "historical_data_for_production_model":  # Mostra info solo per il dataset di training principale
            print("\nInformazioni sul dataset di training:")
            df.info()
        return df
    except FileNotFoundError:
        print(f"ERRORE: Il file '{file_path}' non è stato trovato. Assicurati che sia nella directory corretta.")
        return None
    except Exception as e:
        print(f"ERRORE durante il caricamento del file '{file_path}': {e}")
        return None


def preprocess_data_and_create_target(df: pd.DataFrame, target_col_name: str, features_list: list) -> tuple[
                                                                                                          pd.DataFrame, pd.Series, pd.DataFrame, list] | \
                                                                                                      tuple[
                                                                                                          None, None, None, None]:
    """Esegue il preprocessing (se necessario) e crea la colonna target."""
    print("\n--- Preprocessing Dati Storici e Creazione Target ---")
    df_processed = df.copy()

    # Questo blocco è specifico per il formato di DATA_FILE
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
        print(
            "ATTENZIONE: Colonne 'home_goals' o 'away_goals' non trovate. Necessarie per creare 'original_result' e target per il training.")
        # Se queste colonne non ci sono, non possiamo creare il target per l'addestramento.
        return None, None, None, None

    df_processed[target_col_name] = df_processed['original_result'].apply(lambda x: 1 if x == 'X' else 0)
    print(f"Colonna target '{target_col_name}' creata.")
    print(f"Distribuzione target '{target_col_name}':\n{df_processed[target_col_name].value_counts(normalize=True)}")

    if not all(feature in df_processed.columns for feature in features_list):
        print(f"ERRORE: Feature richieste per il training ({features_list}) non presenti dopo il preprocessing.")
        return None, None, None, None

    X = df_processed[features_list]
    y = df_processed[target_col_name]
    print(f"Feature per il modello: {features_list}")

    categorical_features = [col for col in X.columns if X[col].dtype == 'object' or X[col].dtype.name == 'category']
    print(
        f"Feature categoriche identificate per CatBoost: {categorical_features if categorical_features else 'Nessuna (odds_1 è numerica)'}")

    return X, y, df_processed, categorical_features


def run_stratified_kfold_cv(X_data: pd.DataFrame, y_data: pd.Series, cat_features: list, n_splits: int = 5) -> None:
    """Esegue la K-Fold Cross-Validation Stratificata per valutare la strategia del modello."""
    print(f"\n--- K-Fold Cross-Validation ({n_splits} splits) ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=123)
    fold_metrics = {'accuracy': [], 'auc': [], 'f1_draw': []}

    model_params_cv = dict(
        iterations=1000, learning_rate=0.05, depth=6, l2_leaf_reg=3,
        loss_function='Logloss', eval_metric='AUC', random_seed=42,
        logging_level='Silent', early_stopping_rounds=50, auto_class_weights='Balanced'
    )
    if cat_features: model_params_cv['cat_features'] = cat_features

    for fold, (train_idx, val_idx) in enumerate(skf.split(X_data, y_data)):
        print(f"Fold {fold + 1}/{n_splits}...")
        X_cv_train, X_cv_val = X_data.iloc[train_idx], X_data.iloc[val_idx]
        y_cv_train, y_cv_val = y_data.iloc[train_idx], y_data.iloc[val_idx]

        model_fold = CatBoostClassifier(**model_params_cv)
        model_fold.fit(X_cv_train, y_cv_train, eval_set=(X_cv_val, y_cv_val), verbose=0)

        preds_fold = model_fold.predict(X_cv_val)
        preds_proba_fold = model_fold.predict_proba(X_cv_val)[:, 1]

        fold_metrics['accuracy'].append(accuracy_score(y_cv_val, preds_fold))
        fold_metrics['auc'].append(roc_auc_score(y_cv_val, preds_proba_fold))
        fold_metrics['f1_draw'].append(f1_score(y_cv_val, preds_fold, pos_label=1, zero_division=0))
        print(
            f"  Acc: {fold_metrics['accuracy'][-1]:.4f}, AUC: {fold_metrics['auc'][-1]:.4f}, F1 Pareggio: {fold_metrics['f1_draw'][-1]:.4f}")

    print("\nRisultati medi K-Fold CV:")
    print(f"  Accuratezza media: {np.mean(fold_metrics['accuracy']):.4f} (+/- {np.std(fold_metrics['accuracy']):.4f})")
    print(f"  AUC medio: {np.mean(fold_metrics['auc']):.4f} (+/- {np.std(fold_metrics['auc']):.4f})")
    print(
        f"  F1-Score Pareggio medio: {np.mean(fold_metrics['f1_draw']):.4f} (+/- {np.std(fold_metrics['f1_draw']):.4f})")
    print("--- Fine K-Fold Cross-Validation ---")


def train_catboost_model_with_early_stopping(X_train: pd.DataFrame, y_train: pd.Series,
                                             X_eval_es: pd.DataFrame, y_eval_es: pd.Series,
                                             cat_features: list) -> CatBoostClassifier:
    """Addestra un modello CatBoost con un set di valutazione per l'early stopping."""
    print("\n--- Addestramento Modello CatBoost con Early Stopping ---")
    print(f"Training on {X_train.shape[0]} samples, evaluating for early stopping on {X_eval_es.shape[0]} samples.")

    model_params = dict(
        iterations=1500, learning_rate=0.03, depth=6, l2_leaf_reg=3,
        loss_function='Logloss', eval_metric='AUC', random_seed=4242,  # Seme diverso per questo training specifico
        logging_level='Silent', early_stopping_rounds=100, auto_class_weights='Balanced'
    )
    if cat_features: model_params['cat_features'] = cat_features

    model = CatBoostClassifier(**model_params)
    model.fit(X_train, y_train, eval_set=(X_eval_es, y_eval_es), verbose=500)
    print("Addestramento modello completato.")
    return model


def save_catboost_model(model: CatBoostClassifier, path: str) -> None:
    """Salva il modello CatBoost su disco."""
    print(f"\n--- Salvataggio Modello in '{path}' ---")
    try:
        model.save_model(path)
        print(f"Modello salvato con successo in: {path}")
    except Exception as e:
        print(f"ERRORE durante il salvataggio del modello: {e}")


def load_catboost_model(path: str) -> CatBoostClassifier | None:
    """Carica un modello CatBoost da disco."""
    print(f"\n--- Caricamento Modello da '{path}' ---")
    if not os.path.exists(path):
        print(f"ERRORE: File modello '{path}' non trovato.")
        return None
    try:
        model = CatBoostClassifier()
        model.load_model(path)
        print(f"Modello caricato con successo da: {path}")
        return model
    except Exception as e:
        print(f"ERRORE durante il caricamento del modello da '{path}': {e}")
        return None


def evaluate_model_performance(model: CatBoostClassifier, X_set: pd.DataFrame, y_set: pd.Series, set_name: str) -> dict:
    """Valuta le performance del modello su un dato set (senza salvataggio Excel)."""
    print(f"\n--- Valutazione Performance Modello su Set '{set_name}' ---")
    y_pred = model.predict(X_set)
    y_pred_proba = model.predict_proba(X_set)[:, 1]

    metrics = {
        'accuracy': accuracy_score(y_set, y_pred),
        'auc': roc_auc_score(y_set, y_pred_proba),
        'f1_draw': f1_score(y_set, y_pred, pos_label=1, zero_division=0),
        'classification_report_str': classification_report(y_set, y_pred,
                                                           target_names=['Non Pareggio (0)', 'Pareggio (1)'],
                                                           zero_division=0),
        'confusion_matrix': confusion_matrix(y_set, y_pred)
    }

    print(f"Accuratezza: {metrics['accuracy']:.4f}")
    print(f"AUC: {metrics['auc']:.4f}")
    print(f"F1-Score Pareggio: {metrics['f1_draw']:.4f}")
    print(f"\nClassification Report:\n{metrics['classification_report_str']}")

    print(f"\nMatrice di Confusione:")
    print(metrics['confusion_matrix'])
    plt.figure(figsize=(7, 5))  # Ridimensionato leggermente
    sns.heatmap(metrics['confusion_matrix'], annot=True, fmt='d', cmap='viridis',
                xticklabels=['Non Pareggio', 'Pareggio'], yticklabels=['Non Pareggio', 'Pareggio'])
    plt.xlabel('Etichetta Predetta');
    plt.ylabel('Etichetta Reale')
    plt.title(f'Matrice di Confusione - Set {set_name}')
    plt.tight_layout()
    plt.show()
    return metrics


def display_feature_importance(model: CatBoostClassifier, feature_names: list) -> None:
    """Mostra l'importanza delle feature del modello."""
    print("\n--- Analisi delle Feature Importance (Modello di Produzione) ---")
    if hasattr(model, 'get_feature_importance') and len(feature_names) > 0:
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.get_feature_importance()
        }).sort_values(by='importance', ascending=False)
        print(importance_df)

        if len(feature_names) == 1:  # Specifico per il caso di una sola feature
            print(
                f"Importanza per la feature '{feature_names[0]}': {importance_df['importance'].iloc[0]:.2f}% (CatBoost assegna 100% a una singola feature usata)")
        else:
            plt.figure(figsize=(10, max(2, len(feature_names) * 0.4)))
            sns.barplot(x='importance', y='feature', data=importance_df)
            plt.title(f'Importanza delle Feature (Modello di Produzione)')
            plt.tight_layout()
            plt.show()
    else:
        print("Impossibile mostrare feature importance: metodo non disponibile o nessuna feature.")


def predict_on_new_data_from_file(input_file_path: str, model_path: str, features_list: list,
                                  output_file_path: str | None = None) -> pd.DataFrame | None:
    """Carica nuovi dati, fa predizioni usando un modello salvato e opzionalmente salva i risultati."""
    print(f"\n--- Predizione su Nuovi Dati da File '{input_file_path}' ---")

    new_data_df = load_data(input_file_path, purpose="new_data_for_prediction")
    if new_data_df is None or new_data_df.empty:
        print(f"Nessun dato caricato da '{input_file_path}' o file vuoto. Impossibile predire.")
        return None

    if not all(feature in new_data_df.columns for feature in features_list):
        print(f"ERRORE: I nuovi dati da '{input_file_path}' devono contenere le colonne: {features_list}")
        missing_cols = [f for f in features_list if f not in new_data_df.columns]
        print(f"Colonne mancanti: {missing_cols}")
        return None

    X_new = new_data_df[features_list].copy()

    loaded_model = load_catboost_model(model_path)
    if loaded_model is None:
        return None

    print(f"Effettuazione predizioni sui {X_new.shape[0]} nuovi campioni...")
    try:
        predictions = loaded_model.predict(X_new)
        probabilities = loaded_model.predict_proba(X_new)[:, 1]  # Probabilità per la classe '1' (pareggio)
        print("Predizioni completate.")

        # Aggiungi risultati al DataFrame originale dei nuovi dati
        new_data_df['predicted_is_draw'] = predictions
        new_data_df['probability_of_draw'] = probabilities

        print('\nPrime 5 righe dei risultati delle predizioni sui nuovi dati:')
        print(new_data_df.head())

        if output_file_path:
            try:
                new_data_df.to_csv(output_file_path, index=False)
                print(f"Predizioni sui nuovi dati salvate con successo in: '{output_file_path}'")
            except Exception as e:
                print(f"ERRORE durante il salvataggio dei risultati delle predizioni in '{output_file_path}': {e}")
        return new_data_df

    except Exception as e:
        print(f"ERRORE durante la predizione su nuovi dati: {e}")
        return None


def create_sample_prediction_file_if_not_exists(file_path: str, features_to_include: list):
    """Crea un file CSV di esempio per le predizioni se non esiste già, specificamente per 'odds_1'."""
    if not os.path.exists(file_path):
        print(f"Il file '{file_path}' non esiste. Creazione di un file di esempio...")
        if features_to_include == ['odds_1']:  # Assicura che sia solo per questo caso
            sample_data = {'odds_1': [150, 200, 250, 300, 350, 400, 220, 280, 199, 333]}
            sample_df = pd.DataFrame(sample_data)
            try:
                sample_df.to_csv(file_path, index=False)
                print(f"File di esempio '{file_path}' creato con successo con colonna 'odds_1'.")
            except Exception as e:
                print(f"ERRORE durante la creazione del file di esempio '{file_path}': {e}")
        else:
            # Questo previene la creazione se FEATURES_TO_USE cambia in futuro senza aggiornare questa logica
            print(
                f"ATTENZIONE: La logica di creazione file di esempio è implementata solo per FEATURES_TO_USE=['odds_1']. File non creato.")
    else:
        print(f"Il file '{file_path}' per le nuove predizioni esiste già. Verrà utilizzato quello.")


# --- Flusso Principale dello Script ---
if __name__ == "__main__":
    print("--- INIZIO SCRIPT DI ADDESTRAMENTO MODELLO DI PRODUZIONE E PREDIZIONE ---")

    # STEP 1: Caricamento e preprocessing di TUTTI i dati storici per l'addestramento
    df_historical = load_data(DATA_FILE, purpose="historical_data_for_production_model")
    if df_historical is None:
        print("Script terminato a causa di errore nel caricamento dati storici.")
        exit()

    X_all_data, y_all_data, _, cat_features_for_model = preprocess_data_and_create_target(
        df_historical, TARGET_COLUMN, FEATURES_TO_USE
    )
    if X_all_data is None or y_all_data is None:
        print("Script terminato a causa di errore nel preprocessing dei dati storici.")
        exit()

    # STEP 2: Validazione della strategia del modello e degli iperparametri tramite K-Fold CV su tutti i dati storici
    print("\n\n--- STEP 2: VALIDAZIONE STRATEGIA MODELLO (K-FOLD CV SU DATI STORICI) ---")
    run_stratified_kfold_cv(X_all_data, y_all_data, cat_features_for_model)

    # STEP 3: Addestramento del Modello di Produzione Finale
    # Si usa una porzione dei dati storici (X_all_data, y_all_data) per l'early stopping.
    print("\n\n--- STEP 3: ADDESTRAMENTO MODELLO DI PRODUZIONE FINALE ---")
    X_prod_main_train, X_prod_es_eval, y_prod_main_train, y_prod_es_eval = train_test_split(
        X_all_data, y_all_data, test_size=0.10, random_state=4242, stratify=y_all_data  # 10% per early stopping
    )

    production_model = train_catboost_model_with_early_stopping(
        X_prod_main_train, y_prod_main_train,
        X_prod_es_eval, y_prod_es_eval,
        cat_features_for_model
    )

    # STEP 4: Salvataggio del Modello di Produzione
    save_catboost_model(production_model, MODEL_SAVE_PATH)

    # STEP 5: (Sanity Check) Valutazione del modello di produzione sul suo set interno di early stopping
    print("\n\n--- STEP 5: SANITY CHECK SU SET INTERNO DI EARLY STOPPING DEL MODELLO DI PRODUZIONE ---")
    if production_model:  # Controlla se il modello è stato addestrato
        _ = evaluate_model_performance(production_model, X_prod_es_eval, y_prod_es_eval, "InternalES_ProductionModel")
    else:
        print("Modello di produzione non addestrato, impossibile fare sanity check.")

    # STEP 6: Analisi dell'Importanza delle Feature del Modello di Produzione
    if production_model:
        display_feature_importance(production_model, FEATURES_TO_USE)

    # STEP 7: Predizione su Nuovi Dati da File utilizzando il Modello di Produzione Salvato
    print(f"\n\n--- STEP 7: PREDIZIONE SU NUOVI DATI DAL FILE '{NEW_PREDICTIONS_INPUT_FILE}' ---")
    # Assicura che il file di input per le nuove predizioni esista (crea un esempio se necessario)
    create_sample_prediction_file_if_not_exists(NEW_PREDICTIONS_INPUT_FILE, FEATURES_TO_USE)

    # La funzione predict_on_new_data_from_file carica il modello internamente
    predicted_new_data_df = predict_on_new_data_from_file(
        NEW_PREDICTIONS_INPUT_FILE,
        MODEL_SAVE_PATH,
        FEATURES_TO_USE,
        NEW_PREDICTIONS_OUTPUT_FILE  # Percorso per salvare i risultati
    )

    if predicted_new_data_df is None:
        print(f"Predizione su '{NEW_PREDICTIONS_INPUT_FILE}' non riuscita o file vuoto.")

    # STEP 8: Istruzioni Finali
    print("\n\n--- STEP 8: ISTRUZIONI FINALI ---")
    print(f"Lo script ha completato l'addestramento del modello di produzione, salvato in: {MODEL_SAVE_PATH}")
    print(f"È stata eseguita una K-Fold CV sui dati storici per validare la strategia.")
    print(f"Sono state effettuate predizioni sui dati presenti nel file '{NEW_PREDICTIONS_INPUT_FILE}'.")
    if os.path.exists(NEW_PREDICTIONS_INPUT_FILE):
        print(f"  (Se '{NEW_PREDICTIONS_INPUT_FILE}' non esisteva, ne è stato creato uno di esempio).")
    if predicted_new_data_df is not None and os.path.exists(NEW_PREDICTIONS_OUTPUT_FILE):
        print(
            f"I risultati di queste predizioni sono stati stampati nel log e salvati in '{NEW_PREDICTIONS_OUTPUT_FILE}'.")
    elif predicted_new_data_df is not None:
        print(
            "I risultati di queste predizioni sono stati stampati nel log, ma il salvataggio potrebbe non essere avvenuto.")
    print("\nPer utilizzare il modello in un altro contesto:")
    print(f"1. Assicurati di avere il file modello: '{MODEL_SAVE_PATH}'")
    print(f"2. Prepara un file CSV (es. 'input_quote.csv') con una colonna chiamata '{FEATURES_TO_USE[0]}'.")
    print(
        f"3. Usa la funzione 'predict_on_new_data_from_file' (o adatta 'load_catboost_model' e 'predict_proba') per fare predizioni.")

    print("\n--- FINE DELLO SCRIPT ---")
