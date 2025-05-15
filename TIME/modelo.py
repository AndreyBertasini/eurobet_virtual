import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from collections import deque
import optuna

# --- Impostazioni e Costanti ---
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.4f}'.format)
plt.style.use('seaborn-v0_8-whitegrid')

FILE_PATH = 'model_data.csv'
DATETIME_FORMAT = '%d-%m-%Y %H:%M:%S'
ORIGINAL_TARGET_VARIABLE = 'is_draw'

FEATURES_FOR_SURVIVAL_MODEL = [
    'hour_of_day_sin_start', 'hour_of_day_cos_start',
    'day_of_week_sin_start', 'day_of_week_cos_start',
    'month_sin_start', 'month_cos_start',
    'year_start',
    'is_weekend_start',
    'prev_streak_length',
    'avg_gdsld_last_3_streaks_before_current',
    'std_gdsld_last_3_streaks_before_current',
    'prev_streak_length_x_hour_sin'
]

ORIG_TRAIN_RATIO = 0.70
VALIDATION_SPLIT_FROM_TRAIN_RATIO = 0.20
N_FUTURE_STREAKS_TO_SIMULATE = 10
FUTURE_MATCH_MINUTE_INTERVAL = 5
MAX_LAGS_NEEDED_FOR_FEATURES = 3
OPTUNA_N_TRIALS_SURVIVAL = 70
LONG_NON_DRAW_STREAK_THRESHOLD = 5


# --- Funzioni Helper ---
def _calculate_streak_ending_here(series_is_event_resets_streak):
    streaks = [];
    current_streak = 0
    for is_reset_event in series_is_event_resets_streak:
        if is_reset_event:
            current_streak = 0
        else:
            current_streak += 1
        streaks.append(current_streak)
    return pd.Series(streaks, index=series_is_event_resets_streak.index)


def calculate_rolling_stats_at_draws(df, gdsld_col, draw_col, n=3):
    """Calcola media e std mobile delle lunghezze delle strisce CONCLUSE."""
    gdsld_at_draw_points = df.loc[df[draw_col] == 1, gdsld_col].copy()

    avg_gdsld = gdsld_at_draw_points.rolling(window=n, min_periods=1).mean()
    std_gdsld = gdsld_at_draw_points.rolling(window=n, min_periods=1).std().fillna(0)

    df[f'avg_gdsld_at_draw_time_last{n}'] = np.nan
    df.loc[gdsld_at_draw_points.index, f'avg_gdsld_at_draw_time_last{n}'] = avg_gdsld
    df[f'avg_gdsld_at_draw_time_last{n}'] = df[f'avg_gdsld_at_draw_time_last{n}'].fillna(method='ffill').fillna(0)

    df[f'std_gdsld_at_draw_time_last{n}'] = np.nan
    df.loc[gdsld_at_draw_points.index, f'std_gdsld_at_draw_time_last{n}'] = std_gdsld
    df[f'std_gdsld_at_draw_time_last{n}'] = df[f'std_gdsld_at_draw_time_last{n}'].fillna(method='ffill').fillna(0)

    df[f'avg_gdsld_shifted_for_streak_start_last{n}'] = df[f'avg_gdsld_at_draw_time_last{n}'].shift(1).fillna(0)
    df[f'std_gdsld_shifted_for_streak_start_last{n}'] = df[f'std_gdsld_at_draw_time_last{n}'].shift(1).fillna(0)
    return df


def load_and_preprocess_base_data(file_path, datetime_format):
    print(f"Caricamento dati da: {file_path}")
    try:
        df = pd.read_csv(file_path, dtype=str)
    except Exception as e:
        print(f"Errore caricamento: {e}"); return None
    if not all(col in df.columns for col in ['date', 'hour']): print(
        "Errore: Colonne 'date'/'hour' mancanti."); return None

    df['datetime'] = pd.to_datetime(df['date'].str.strip() + ' ' + df['hour'].str.strip(), format=datetime_format,
                                    errors='coerce')
    df.dropna(subset=['datetime'], inplace=True)
    if df.empty: print("DataFrame vuoto dopo rimozione datetime non validi."); return None
    df = df.sort_values('datetime').reset_index(drop=True)

    cols_goals = ['home_goals', 'away_goals']
    for col in cols_goals: df[col] = pd.to_numeric(df.get(col), errors='coerce')
    df.dropna(subset=cols_goals, inplace=True)
    if df.empty: print("DataFrame vuoto dopo pulizia gol."); return None

    df[ORIGINAL_TARGET_VARIABLE] = (df['home_goals'] == df['away_goals']).astype(int)
    df['_temp_streak'] = _calculate_streak_ending_here(df[ORIGINAL_TARGET_VARIABLE])
    df['global_distance_since_last_draw'] = df['_temp_streak'].shift(1).fillna(0)
    df.drop(columns=['_temp_streak'], inplace=True)

    df = calculate_rolling_stats_at_draws(df, 'global_distance_since_last_draw', ORIGINAL_TARGET_VARIABLE,
                                          n=MAX_LAGS_NEEDED_FOR_FEATURES)

    print("Pre-elaborazione dati base completata.")
    return df


def prepare_survival_data(df_processed):
    print("Preparazione dati per Survival Analysis...")
    draw_events_df = df_processed[df_processed[ORIGINAL_TARGET_VARIABLE] == 1].copy()
    survival_data_list = []

    for current_draw_original_idx in draw_events_df.index:
        duration = df_processed.loc[current_draw_original_idx, 'global_distance_since_last_draw']
        if duration <= 0: continue

        start_of_current_streak_original_idx = current_draw_original_idx - int(duration)
        if start_of_current_streak_original_idx < 0: continue

        dt_features_provider = df_processed.loc[start_of_current_streak_original_idx, 'datetime']
        idx_of_previous_draw_event = start_of_current_streak_original_idx - 1

        prev_streak_len = 0
        avg_prev_3_streaks = 0
        std_prev_3_streaks = 0

        if idx_of_previous_draw_event >= 0:
            if df_processed.loc[idx_of_previous_draw_event, ORIGINAL_TARGET_VARIABLE] == 1:
                prev_streak_len = df_processed.loc[idx_of_previous_draw_event, 'global_distance_since_last_draw']
                avg_prev_3_streaks = df_processed.loc[
                    idx_of_previous_draw_event, f'avg_gdsld_at_draw_time_last{MAX_LAGS_NEEDED_FOR_FEATURES}']
                std_prev_3_streaks = df_processed.loc[
                    idx_of_previous_draw_event, f'std_gdsld_at_draw_time_last{MAX_LAGS_NEEDED_FOR_FEATURES}']

        hour_sin_start_val = np.sin(2 * np.pi * dt_features_provider.hour / 24.0)

        survival_data_list.append({
            'duration': duration, 'event': 1,
            'hour_of_day_sin_start': hour_sin_start_val,
            'hour_of_day_cos_start': np.cos(2 * np.pi * dt_features_provider.hour / 24.0),
            'day_of_week_sin_start': np.sin(2 * np.pi * dt_features_provider.dayofweek / 7.0),
            'day_of_week_cos_start': np.cos(2 * np.pi * dt_features_provider.dayofweek / 7.0),
            'month_sin_start': np.sin(2 * np.pi * dt_features_provider.month / 12.0),
            'month_cos_start': np.cos(2 * np.pi * dt_features_provider.month / 12.0),
            'year_start': dt_features_provider.year,
            'is_weekend_start': int(dt_features_provider.dayofweek >= 5),
            'prev_streak_length': prev_streak_len,
            'avg_gdsld_last_3_streaks_before_current': avg_prev_3_streaks,
            'std_gdsld_last_3_streaks_before_current': std_prev_3_streaks,
            'prev_streak_length_x_hour_sin': prev_streak_len * hour_sin_start_val,
            'original_event_time': df_processed.loc[current_draw_original_idx, 'datetime']
        })

    df_survival = pd.DataFrame(survival_data_list)
    if df_survival.empty: print("Nessun dato di sopravvivenza generato."); return None

    df_survival = df_survival.sort_values(by='original_event_time').reset_index(drop=True)
    print(f"Dati di sopravvivenza preparati. Righe: {len(df_survival)}")
    print(df_survival.head())
    return df_survival


def objective_survival_cox(trial, X_train, y_train_duration, y_train_event, X_val, y_val_duration, y_val_event):
    df_train_lifelines = X_train.copy();
    df_train_lifelines['duration'] = y_train_duration;
    df_train_lifelines['event'] = y_train_event
    df_val_lifelines = X_val.copy();
    df_val_lifelines['duration'] = y_val_duration;
    df_val_lifelines['event'] = y_val_event
    penalizer = trial.suggest_float('penalizer', 1e-6, 2.0, log=True)
    cph = CoxPHFitter(penalizer=penalizer)
    try:
        cph.fit(df_train_lifelines, duration_col='duration', event_col='event', show_progress=False)
        if not df_val_lifelines.empty:
            c_index = cph.score(df_val_lifelines, scoring_method="concordance_index")
        else:
            c_index = 0.5
    except Exception as e:
        print(f"Trial {trial.number} fallito CoxPH: {e}");
        return 0.0
    return c_index


def train_final_survival_model(best_params, X_train, y_train_duration, y_train_event, X_val, y_val_duration,
                               y_val_event):
    print("\n--- Addestramento Modello Finale CoxPHFitter ---")
    df_train_lifelines = X_train.copy();
    df_train_lifelines['duration'] = y_train_duration;
    df_train_lifelines['event'] = y_train_event
    df_val_lifelines = X_val.copy();
    df_val_lifelines['duration'] = y_val_duration;
    df_val_lifelines['event'] = y_val_event
    final_penalizer = best_params.get('penalizer', 0.1)
    cph = CoxPHFitter(penalizer=final_penalizer)
    try:
        cph.fit(df_train_lifelines, duration_col='duration', event_col='event', show_progress=True)
        print("Modello Finale CoxPHFitter addestrato.")
        fitted = True
        if not df_val_lifelines.empty:
            val_c_index = cph.score(df_val_lifelines, scoring_method="concordance_index")
            print(f"C-index su Validation Set (modello finale): {val_c_index:.4f}")
    except Exception as e:
        print(f"Errore addestramento finale CoxPH: {e}"); return None, False
    return cph, fitted


def evaluate_survival_model_on_test(model, X_test, y_test_duration, y_test_event):
    print("\n--- Valutazione Modello di Sopravvivenza su Test Set ---")
    df_test_lifelines = X_test.copy();
    df_test_lifelines['duration'] = y_test_duration;
    df_test_lifelines['event'] = y_test_event
    if model is None or df_test_lifelines.empty: print("Modello non addestrato o test set vuoto."); return
    test_c_index = model.score(df_test_lifelines, scoring_method="concordance_index")
    print(f"C-index su Test Set: {test_c_index:.4f}")
    print("\nCoefficienti del Modello (Hazards Ratios):");
    model.print_summary(decimals=4, model="Cox Proportional Hazards Model")
    plt.figure(figsize=(10, max(6, len(model.params_) * 0.5)));
    model.plot()
    plt.title("Coefficienti del Modello CoxPH (log(Hazard Ratio))");
    plt.show()


def check_proportional_hazards_assumption(model, X_test_df, y_test_duration_series, y_test_event_series):
    """Verifica l'assunzione di rischi proporzionali sul test set."""
    print("\n--- Verifica Assunzione Rischi Proporzionali (Test Set) ---")
    if model is None: print("Modello non addestrato, impossibile verificare le assunzioni."); return
    df_check = X_test_df.copy()
    df_check['duration'] = y_test_duration_series
    df_check['event'] = y_test_event_series
    if df_check.empty: print("Dati per il check delle assunzioni vuoti."); return
    try:
        model.check_assumptions(df_check, p_value_threshold=0.05, show_plots=True)
        print("Verifica delle assunzioni completata. Controllare l'output e i grafici sopra.")
    except Exception as e:
        print(f"Errore durante la verifica delle assunzioni: {e}")
        print("Potrebbe essere dovuto a dati insufficienti o problemi di collinearità residua.")


def simulate_future_streaks_survival(cox_model, last_observed_streak_data, historic_observed_streaks, num_sim_streaks,
                                     interval_minutes, features_list_survival, max_lags_hist_for_avg_param):
    print("\n--- Simulazione Distanze Future tra Pareggi con Modello di Sopravvivenza ---")
    simulated_streak_lengths = []
    current_sim_time = last_observed_streak_data['original_event_time']
    current_prev_streak_length = last_observed_streak_data['duration']
    historic_streaks_for_stats = deque(historic_observed_streaks, maxlen=int(max_lags_hist_for_avg_param))
    long_streak_signal_in_sim = False

    for i in range(num_sim_streaks):
        current_sim_time += timedelta(minutes=interval_minutes)
        current_avg_gdsld_for_feat = np.mean(historic_streaks_for_stats) if historic_streaks_for_stats else 0
        current_std_gdsld_for_feat = np.std(historic_streaks_for_stats) if len(historic_streaks_for_stats) > 1 else 0
        hour_sin_start_val = np.sin(2 * np.pi * current_sim_time.hour / 24.0)
        features_for_prediction_dict = {
            'hour_of_day_sin_start': hour_sin_start_val,
            'hour_of_day_cos_start': np.cos(2 * np.pi * current_sim_time.hour / 24.0),
            'day_of_week_sin_start': np.sin(2 * np.pi * current_sim_time.dayofweek / 7.0),
            'day_of_week_cos_start': np.cos(2 * np.pi * current_sim_time.dayofweek / 7.0),
            'month_sin_start': np.sin(2 * np.pi * current_sim_time.month / 12.0),
            'month_cos_start': np.cos(2 * np.pi * current_sim_time.month / 12.0),
            'year_start': current_sim_time.year,
            'is_weekend_start': int(current_sim_time.dayofweek >= 5),
            'prev_streak_length': current_prev_streak_length,
            'avg_gdsld_last_3_streaks_before_current': current_avg_gdsld_for_feat,
            'std_gdsld_last_3_streaks_before_current': current_std_gdsld_for_feat,
            'prev_streak_length_x_hour_sin': current_prev_streak_length * hour_sin_start_val
        }
        current_X_sim_df = pd.DataFrame(
            {feat: [features_for_prediction_dict.get(feat, 0)] for feat in features_list_survival}, index=[0])
        prediction_result = cox_model.predict_median(current_X_sim_df)
        predicted_median_val = 0.0
        if isinstance(prediction_result, (pd.Series, pd.DataFrame)):
            if not prediction_result.empty:
                predicted_median_val = prediction_result.iloc[0]
            else:
                predicted_median_val = np.nan
        elif isinstance(prediction_result, (float, np.float64, int, np.int64)):
            predicted_median_val = prediction_result
        else:
            predicted_median_val = np.nan
        if pd.isna(predicted_median_val) or np.isinf(predicted_median_val):
            print(f"Simulazione {i + 1}: Predizione mediana non valida, uso fallback (3).")
            predicted_duration = 3
        else:
            predicted_duration = max(1, int(round(predicted_median_val)))
        simulated_streak_lengths.append(predicted_duration)
        if predicted_duration > LONG_NON_DRAW_STREAK_THRESHOLD: long_streak_signal_in_sim = True
        current_prev_streak_length = predicted_duration
        historic_streaks_for_stats.append(predicted_duration)
        current_sim_time += timedelta(minutes=predicted_duration * FUTURE_MATCH_MINUTE_INTERVAL)
    print("\nDistanze tra Pareggi Predette dalla Simulazione:")
    print(simulated_streak_lengths)
    return simulated_streak_lengths, long_streak_signal_in_sim


def generate_survival_report(simulated_lengths, long_streak_alert):
    print("\n--- Report Predizioni Distanze (Survival Analysis) ---")
    if not simulated_lengths: print("Nessuna distanza simulata."); return
    print("\n1. Distanze tra Pareggi Predette (lunghezze delle strisce di non-pareggio simulate):")
    print(simulated_lengths)
    if simulated_lengths:
        print(
            f"  - Media: {np.mean(simulated_lengths):.2f}, Max: {np.max(simulated_lengths)}, Min: {np.min(simulated_lengths)}")
    print(f"\n2. Analisi Tendenza Distanze Lunghe (>{LONG_NON_DRAW_STREAK_THRESHOLD} non-pareggi consecutivi):")
    if long_streak_alert:
        num_long_streaks_pred = sum(s > LONG_NON_DRAW_STREAK_THRESHOLD for s in simulated_lengths)
        print(f"  - SEGNALE: Prevista almeno una striscia di non-pareggi >{LONG_NON_DRAW_STREAK_THRESHOLD} partite.")
        print(f"  - Numero di tali strisce predette: {num_long_streaks_pred}")
    else:
        print(f"  - NESSUN segnale chiaro di strisce >{LONG_NON_DRAW_STREAK_THRESHOLD} partite nella simulazione.")
    print("--- Fine Report Survival Analysis ---")


# --- Flusso Principale ---
def main():
    print("Ciao Andrea! Predizione DISTANZA TRA PAREGGI con Survival Analysis (CoxPH, Feature Eng V2, Optuna).")

    df_base = load_and_preprocess_base_data(FILE_PATH, DATETIME_FORMAT)
    if df_base is None or df_base.empty: print("Terminazione: errori preparazione dati base."); return

    df_survival = prepare_survival_data(df_base)
    if df_survival is None or df_survival.empty: print("Terminazione: errori preparazione dati sopravvivenza."); return

    df_survival_ready = df_survival.dropna(subset=FEATURES_FOR_SURVIVAL_MODEL + ['duration', 'event']).copy()
    if df_survival_ready.empty: print("DataFrame sopravvivenza vuoto dopo rimozione NaN."); return

    print(f"Campioni per modello sopravvivenza: {len(df_survival_ready)}")
    print(f"Statistiche descrittive per 'duration':\n{df_survival_ready['duration'].describe()}")

    train_val_indices, test_indices = train_test_split(df_survival_ready.index, test_size=(1 - ORIG_TRAIN_RATIO),
                                                       shuffle=False)
    df_train_val = df_survival_ready.loc[train_val_indices];
    df_test = df_survival_ready.loc[test_indices]
    train_indices, val_indices = train_test_split(df_train_val.index, test_size=VALIDATION_SPLIT_FROM_TRAIN_RATIO,
                                                  shuffle=False)
    df_train = df_train_val.loc[train_indices];
    df_val = df_train_val.loc[val_indices]

    X_train = df_train[FEATURES_FOR_SURVIVAL_MODEL];
    y_train_duration = df_train['duration'];
    y_train_event = df_train['event']
    X_val = df_val[FEATURES_FOR_SURVIVAL_MODEL];
    y_val_duration = df_val['duration'];
    y_val_event = df_val['event']
    X_test = df_test[FEATURES_FOR_SURVIVAL_MODEL];
    y_test_duration = df_test['duration'];
    y_test_event = df_test['event']

    if any(s.empty for s in [X_train, X_val, X_test, y_train_duration, y_val_duration, y_test_duration]):
        print("Errore: Uno dei set (Train, Validation, Test) è vuoto.");
        return
    print(f"Dimensioni: Train=({X_train.shape}), Validation=({X_val.shape}), Test=({X_test.shape})")

    study_survival = optuna.create_study(direction='maximize')
    try:
        study_survival.optimize(
            lambda trial: objective_survival_cox(trial, X_train, y_train_duration, y_train_event, X_val, y_val_duration,
                                                 y_val_event),
            n_trials=OPTUNA_N_TRIALS_SURVIVAL)
        print("\nMigliori iperparametri (CoxPH):", study_survival.best_params)
        print(f"Miglior C-index su validation (Optuna): {study_survival.best_value:.4f}")
        best_params_cox = study_survival.best_params
    except Exception as e:
        print(f"Errore Optuna CoxPH: {e}");
        best_params_cox = {'penalizer': 0.1}

    model_cox, fitted_cox = train_final_survival_model(  # Assegna a model_cox
        best_params_cox, X_train, y_train_duration, y_train_event, X_val, y_val_duration, y_val_event
    )

    if fitted_cox:
        evaluate_survival_model_on_test(model_cox, X_test, y_test_duration, y_test_event)  # Usa model_cox
        check_proportional_hazards_assumption(model_cox, X_test, y_test_duration, y_test_event)  # Usa model_cox

        if not df_survival_ready.empty:
            last_observed_streak_data_dict = df_survival_ready.iloc[-1].to_dict()
            initial_historic_streaks = df_survival_ready['duration'].tail(MAX_LAGS_NEEDED_FOR_FEATURES).tolist()

            # CORREZIONE: Nome variabile corretto
            sim_lengths, long_alert = simulate_future_streaks_survival(
                model_cox,  # Usa model_cox
                last_observed_streak_data_dict,
                initial_historic_streaks,
                N_FUTURE_STREAKS_TO_SIMULATE,
                FUTURE_MATCH_MINUTE_INTERVAL,
                FEATURES_FOR_SURVIVAL_MODEL,
                MAX_LAGS_NEEDED_FOR_FEATURES
            )
            generate_survival_report(sim_lengths, long_alert)
        else:
            print("Non ci sono dati osservati per iniziare la simulazione futura.")
    else:
        print("Simulazione futura skippata.")
    print("\n--- Fine dell'analisi. Ciao Andrea! ---")


if __name__ == '__main__':
    main()
