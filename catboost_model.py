import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns

# Impostazioni per una migliore visualizzazione dell'output di pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# --- 1. Caricamento Dati ---
try:
    df = pd.read_csv('virtual_matches_data.csv')
except FileNotFoundError:
    print(
        "Errore: Il file 'virtual_matches_data.csv' non è stato trovato. Assicurati che sia nella directory corretta.")
    exit()

print("Prime 5 righe del dataset:")
print(df.head())
print("\nInformazioni sul dataset:")
df.info()

# --- 2. Preprocessing e Feature Engineering ---

# --- Creazione della variabile 'original_result' (1, X, 2) per identificare i pareggi ---
if 'home_goals' in df.columns and 'away_goals' in df.columns:
    def determine_original_result(row):
        if pd.isna(row['home_goals']) or pd.isna(row['away_goals']):
            return np.nan
        if row['home_goals'] > row['away_goals']:
            return '1'
        elif row['home_goals'] < row['away_goals']:
            return '2'
        else:
            return 'X'


    df['original_result'] = df.apply(determine_original_result, axis=1)
    df.dropna(subset=['original_result'], inplace=True)
else:
    print("Errore: Le colonne 'home_goals' o 'away_goals' necessarie per creare 'original_result' non sono presenti.")
    exit()

# --- Creazione della variabile target binaria 'is_draw' ---
df['is_draw'] = df['original_result'].apply(lambda x: 1 if x == 'X' else 0)
print("\nDistribuzione della variabile target 'is_draw':")
print(df['is_draw'].value_counts(normalize=True))

# --- Selezione delle Feature (X) e del Target (y) ---
# MODIFICATO: Utilizziamo solo 'odds_1' come feature
features_to_use = ['odds_1']

# Verifica che la feature 'odds_1' esista
if 'odds_1' not in df.columns:
    print("Errore: La colonna 'odds_1' non è presente nel DataFrame. Impossibile procedere.")
    exit()

print(f"\nFeature selezionate per il modello: {features_to_use}")

X = df[features_to_use]
y = df['is_draw']

# --- Identificazione delle Feature Categoriche per CatBoost ---
# Con solo 'odds_1' (tipicamente numerica), questa lista sarà probabilmente vuota.
# CatBoost gestisce automaticamente le feature numeriche.
categorical_features = []
for col in X.columns:  # Il loop ora scorrerà solo su ['odds_1']
    if X[col].dtype == 'object' or X[col].dtype.name == 'category':
        categorical_features.append(col)
# Non è necessario aggiungere altre feature categoriche manualmente qui perché non le stiamo usando.
print(f"\nFeature categoriche identificate per CatBoost: {categorical_features}")

# --- 3. Divisione dei Dati ---
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(
    f"\nDimensioni dei set per CV e training finale: X_train_val {X_train_val.shape}, y_train_val {y_train_val.shape}")
print(f"Dimensioni del Test Set finale: X_test {X_test.shape}, y_test {y_test.shape}")

# --- 4. K-Fold Cross-Validation ---
print("\n--- Inizio Cross-Validation Stratificata (K-Fold) ---")
N_SPLITS = 5
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=123)

fold_accuracies = []
fold_auc_scores = []
fold_f1_scores_draw = []

model_params_cv = dict(
    iterations=1000,
    learning_rate=0.05,
    depth=6,  # Potrebbe essere ridotta per un modello con una sola feature
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    logging_level='Silent',
    early_stopping_rounds=50,
    auto_class_weights='Balanced'
    # cat_features: se categorical_features è vuota, non è necessario passarla esplicitamente
)

for fold, (train_idx, val_idx) in enumerate(skf.split(X_train_val, y_train_val)):
    print(f"Fold {fold + 1}/{N_SPLITS}")
    X_cv_train, X_cv_val = X_train_val.iloc[train_idx], X_train_val.iloc[val_idx]
    y_cv_train, y_cv_val = y_train_val.iloc[train_idx], y_train_val.iloc[val_idx]

    # current_fold_cat_features sarà vuota se 'odds_1' è numerica
    current_fold_cat_features = [col for col in categorical_features if col in X_cv_train.columns]

    model_fold_params = model_params_cv.copy()
    if current_fold_cat_features:
        model_fold_params['cat_features'] = current_fold_cat_features

    model_fold = CatBoostClassifier(**model_fold_params)

    model_fold.fit(
        X_cv_train, y_cv_train,
        eval_set=(X_cv_val, y_cv_val),
        verbose=0
    )
    preds_fold = model_fold.predict(X_cv_val)
    preds_proba_fold = model_fold.predict_proba(X_cv_val)[:, 1]

    acc_fold = accuracy_score(y_cv_val, preds_fold)
    auc_fold = roc_auc_score(y_cv_val, preds_proba_fold)
    f1_draw_fold = f1_score(y_cv_val, preds_fold, pos_label=1, zero_division=0)

    fold_accuracies.append(acc_fold)
    fold_auc_scores.append(auc_fold)
    fold_f1_scores_draw.append(f1_draw_fold)

    print(f"Accuratezza Fold {fold + 1}: {acc_fold:.4f}, AUC: {auc_fold:.4f}, F1-Score Pareggio: {f1_draw_fold:.4f}")

print(f"\nAccuratezza media K-Fold: {np.mean(fold_accuracies):.4f} (+/- {np.std(fold_accuracies):.4f})")
print(f"AUC medio K-Fold: {np.mean(fold_auc_scores):.4f} (+/- {np.std(fold_auc_scores):.4f})")
print(f"F1-Score Pareggio medio K-Fold: {np.mean(fold_f1_scores_draw):.4f} (+/- {np.std(fold_f1_scores_draw):.4f})")
print("--- Fine Cross-Validation ---\n")

# --- 5. Addestramento del Modello Finale e Valutazione sul Test Set ---
X_train_final, X_val_final, y_train_final, y_val_final = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42, stratify=y_train_val
)

print(
    f"Dimensioni per l'addestramento finale: X_train_final {X_train_final.shape}, y_train_final {y_train_final.shape}")
print(
    f"Dimensioni per la validazione finale (early stopping): X_val_final {X_val_final.shape}, y_val_final {y_val_final.shape}")

print("\nInizio addestramento del modello CatBoost finale...")
# final_model_cat_features sarà vuota se 'odds_1' è numerica
final_model_cat_features = [col for col in categorical_features if col in X_train_final.columns]

final_model_params = dict(
    iterations=1500,
    learning_rate=0.03,
    depth=6,  # Considerare di ridurre la profondità per un modello con una sola feature
    l2_leaf_reg=3,
    loss_function='Logloss',
    eval_metric='AUC',
    random_seed=42,
    logging_level='Silent',
    early_stopping_rounds=100,
    auto_class_weights='Balanced'
)
if final_model_cat_features:
    final_model_params['cat_features'] = final_model_cat_features

model_final = CatBoostClassifier(**final_model_params)

model_final.fit(
    X_train_final, y_train_final,
    eval_set=(X_val_final, y_val_final),
    verbose=500
)

# --- 6. Valutazione del Modello Finale sul Test Set ---
print("\nValutazione del modello finale sul Test Set:")
y_pred_test = model_final.predict(X_test)
y_pred_proba_test = model_final.predict_proba(X_test)[:, 1]

accuracy_test = accuracy_score(y_test, y_pred_test)
auc_test = roc_auc_score(y_test, y_pred_proba_test)
f1_draw_test = f1_score(y_test, y_pred_test, pos_label=1, zero_division=0)

print(f"Accuratezza sul Test Set: {accuracy_test:.4f}")
print(f"AUC sul Test Set: {auc_test:.4f}")
print(f"F1-Score Pareggio sul Test Set: {f1_draw_test:.4f}")

print("\nClassification Report sul Test Set (Target: is_draw):")
target_names_binary = ['Non Pareggio (0)', 'Pareggio (1)']
print(classification_report(y_test, y_pred_test, target_names=target_names_binary, zero_division=0))

print("\nMatrice di Confusione sul Test Set (Target: is_draw):")
conf_matrix = confusion_matrix(y_test, y_pred_test)  # Etichette di default [0, 1]
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Non Pareggio', 'Pareggio'], yticklabels=['Non Pareggio', 'Pareggio'])
plt.xlabel('Etichetta Predetta')
plt.ylabel('Etichetta Reale')
plt.title('Matrice di Confusione (Target: is_draw)')
plt.show()

# --- Log Dettagliato dei Pareggi ---
# conf_matrix[0,0] = TN (Non Pareggio predetto Non Pareggio)
# conf_matrix[0,1] = FP (Non Pareggio predetto Pareggio - Falsi Allarmi)
# conf_matrix[1,0] = FN (Pareggio predetto Non Pareggio - Pareggi Mancati)
# conf_matrix[1,1] = TP (Pareggio predetto Pareggio - Pareggi Corretti)
TN = conf_matrix[0, 0]
FP = conf_matrix[0, 1]
FN = conf_matrix[1, 0]
TP = conf_matrix[1, 1]

print("\n--- Dettaglio Predizione Pareggi sul Test Set ---")
print(f"Numero totale di partite nel Test Set: {len(y_test)}")
print(f"Numero di pareggi effettivi nel Test Set: {np.sum(y_test)}")
print(f"Numero di non-pareggi effettivi nel Test Set: {len(y_test) - np.sum(y_test)}")
print("--------------------------------------------------")
print(f"Pareggi effettivi previsti correttamente (TP): {TP}")
print(f"Pareggi effettivi previsti erroneamente come 'Non Pareggio' (FN - Pareggi Mancati): {FN}")
print(f"Non-pareggi effettivi previsti erroneamente come 'Pareggio' (FP - Falsi Allarmi): {FP}")
print(f"Non-pareggi effettivi previsti correttamente (TN): {TN}")
print("--------------------------------------------------")

# --- 7. Analisi delle Feature Importance (Modello Finale) ---
if hasattr(model_final, 'get_feature_importance'):
    print("\nFeature Importance (Modello Finale):")
    # Se X_train_final ha una sola colonna, get_feature_importance restituirà un array con un solo valore.
    # Dobbiamo assicurarci che X_train_final.columns sia usato correttamente.
    if X_train_final.shape[1] > 0:  # Verifica che ci siano colonne
        feature_importance_df = pd.DataFrame({
            'feature': X_train_final.columns,
            'importance': model_final.get_feature_importance()
        })
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False)
        print(feature_importance_df)

        plt.figure(figsize=(10, max(2, len(X_train_final.columns) * 0.3)))  # Adatta altezza per poche feature
        sns.barplot(x='importance', y='feature', data=feature_importance_df)
        plt.title('Importanza delle Feature (CatBoost - Predizione Pareggio con solo odds_1)')
        plt.tight_layout()
        plt.show()
    elif X_train_final.shape[1] == 1:  # Caso specifico per una sola feature
        print(
            f"Importanza per la feature '{X_train_final.columns[0]}': {model_final.get_feature_importance()[0]:.2f}% (approssimativo, CatBoost potrebbe dare 100%)")
    else:
        print("Nessuna feature da mostrare per l'importanza.")

else:
    print("Attenzione: Il metodo get_feature_importance non è disponibile per il modello finale.")

print("\n--- Fine del processo ---")
