import customtkinter
import pandas as pd
from catboost import CatBoostClassifier
import os
import numpy as np
import re  # Importato per il parsing più flessibile

# --- Costanti ---
BASE_MODEL_PATH = r'C:\Users\dyshkantiuk_andrii\Desktop\eurobet_virtual_2\PREDICT_RESULT'
MODEL_FILENAME = "catboost_multiclass_predictor.cbm"
FULL_MODEL_PATH = os.path.join(BASE_MODEL_PATH, MODEL_FILENAME)

# Mappatura da classe predetta (0, 1, 2) a etichetta visualizzata
# Questo dipende da come LabelEncoder ha mappato le tue classi '1', 'X', '2'
# Se le.classes_ era ['1', '2', 'X'] -> {0: '1', 1: '2', 2: 'X'}
# Se le.classes_ era ['1', 'X', '2'] -> {0: '1', 1: 'X', 2: '2'}
# L'output precedente indicava: TARGET_LABELS = {0: '1', 1: '2', 2: 'X'}
# Quindi: Classe 0 -> '1', Classe 1 -> '2', Classe 2 -> 'X'
PREDICTION_LABEL_MAP = {
    0: '1 (Vittoria Casa)',
    1: '2 (Vittoria Ospite)',  # Corrisponde alla classe 1 se le.classes_ = ['1', '2', 'X']
    2: 'X (Pareggio)'  # Corrisponde alla classe 2 se le.classes_ = ['1', '2', 'X']
}


# Le probabilità da predict_proba sono nell'ordine di self.model.classes_

class MulticlassPredictorApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Predittore Risultati Partite (Multi-Input)")
        self.geometry("700x650")
        customtkinter.set_appearance_mode("System")
        customtkinter.set_default_color_theme("blue")

        self.model = None
        self.model_loaded_successfully = False
        self.expected_features = []
        self.model_classes_order_for_proba = None  # Per memorizzare l'ordine delle classi del modello

        self._load_model()
        self._create_widgets()

    def _load_model(self):
        if not os.path.exists(FULL_MODEL_PATH):
            print(f"ERRORE: File modello '{FULL_MODEL_PATH}' non trovato.")
            return

        try:
            self.model = CatBoostClassifier()
            self.model.load_model(FULL_MODEL_PATH)
            self.model_loaded_successfully = True
            if hasattr(self.model, 'feature_names_'):
                self.expected_features = self.model.feature_names_
            else:
                print("ATTENZIONE: Impossibile determinare le feature attese dal modello.")
                self.expected_features = []  # Lascia vuoto se non può essere determinato

            if hasattr(self.model, 'classes_'):
                self.model_classes_order_for_proba = list(self.model.classes_)
                print(f"Classi del modello (ordine per predict_proba): {self.model_classes_order_for_proba}")
            else:
                print("ATTENZIONE: Impossibile determinare l'ordine delle classi dal modello per le probabilità.")

            print(f"Modello '{MODEL_FILENAME}' caricato con successo.")
            print(f"Feature attese dal modello: {self.expected_features}")
        except Exception as e:
            print(f"ERRORE durante il caricamento del modello '{FULL_MODEL_PATH}': {e}")
            self.model = None
            self.model_loaded_successfully = False

    def _create_widgets(self):
        main_frame = customtkinter.CTkFrame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        app_title_label = customtkinter.CTkLabel(main_frame, text="Predittore Risultati Partite (Multi-Input)",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        app_title_label.pack(pady=(10, 15))

        # Istruzioni per l'input
        input_instructions_text = "Inserisci i dati per una o più partite (una partita per riga).\n"
        if self.expected_features:
            input_instructions_text += f"Per ogni riga, inserisci le quote separate da virgola o spazio nell'ordine: {', '.join(self.expected_features)}.\n"
            input_instructions_text += f"Esempio per una riga (se le features sono {len(self.expected_features)}):\n"
            if len(self.expected_features) == 1:
                input_instructions_text += "2.50"
            elif len(self.expected_features) == 3:  # Es. odds_1, odds_X, odds_2
                input_instructions_text += "2.50, 3.20, 2.80  o  250, 320, 280"
            else:  # Generico
                input_instructions_text += ", ".join(["quota"] * len(self.expected_features))

        else:
            input_instructions_text += "Modello non caricato o feature non determinate. Impossibile fornire formato di input."

        input_instructions_label = customtkinter.CTkLabel(main_frame, text=input_instructions_text, justify="left")
        input_instructions_label.pack(pady=(0, 5), anchor="w")

        self.odds_input_textbox = customtkinter.CTkTextbox(main_frame, height=150, width=400)
        self.odds_input_textbox.pack(pady=5, fill="x", expand=True)
        if self.expected_features and len(self.expected_features) == 3:  # Placeholder specifico se 3 features
            self.odds_input_textbox.insert("1.0", "2.10, 3.00, 3.50\n1.80, 3.20, 4.00\n300, 330, 220")
        elif self.expected_features and len(self.expected_features) == 1:
            self.odds_input_textbox.insert("1.0", "2.10\n1.80\n300")

        self.predict_button = customtkinter.CTkButton(main_frame, text="Predici Risultati",
                                                      command=self._on_predict_button_click)
        self.predict_button.pack(pady=15)
        if not self.model_loaded_successfully or not self.expected_features:
            self.predict_button.configure(state="disabled")

        results_title_label = customtkinter.CTkLabel(main_frame, text="Risultati Predetti:",
                                                     font=customtkinter.CTkFont(size=16, weight="bold"))
        results_title_label.pack(pady=(10, 2))
        self.results_display_textbox = customtkinter.CTkTextbox(main_frame, height=200, state="disabled", wrap="word")
        self.results_display_textbox.pack(pady=5, fill="both", expand=True)

        self.status_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=10))
        self.status_label.pack(side="bottom", pady=5, fill="x")

        if not self.model_loaded_successfully:
            self.status_label.configure(text=f"Errore: Modello '{MODEL_FILENAME}' non trovato o corrotto.",
                                        text_color="red")
            self.results_display_textbox.configure(state="normal")
            self.results_display_textbox.insert("1.0", "Modello non caricato.")
            self.results_display_textbox.configure(state="disabled")
        elif not self.expected_features:
            self.status_label.configure(
                text="Modello caricato, ma non è stato possibile determinare le feature attese.", text_color="orange")
        else:
            self.status_label.configure(text=f"Modello '{MODEL_FILENAME}' caricato. Pronto per l'input.",
                                        text_color="green")

    def _parse_odds_line(self, line_str: str, num_expected_features: int) -> list[float] | None:
        """Analizza una singola riga di input per estrarre e normalizzare le quote."""
        # Rimuove spazi extra e splitta per virgola o spazio
        parts = [p.strip() for p in re.split(r'[,;\s]+', line_str) if p.strip()]

        if len(parts) != num_expected_features:
            return None  # Numero errato di quote

        processed_odds = []
        for part_str in parts:
            try:
                value_float = float(part_str.replace(',', '.'))  # Gestisce sia . che , come decimale
                # Normalizzazione: se l'utente inserisce un intero > 20 (es. 250), lo converte (2.50)
                if value_float > 20.0 and value_float == int(value_float):
                    value_float /= 100.0
                processed_odds.append(value_float)
            except ValueError:
                return None  # Valore non numerico
        return processed_odds

    def _on_predict_button_click(self):
        if not self.model_loaded_successfully or not self.expected_features:
            self.status_label.configure(text="Errore: Modello non pronto o feature non definite.", text_color="red")
            return

        all_input_text = self.odds_input_textbox.get("1.0", "end-1c").strip()
        if not all_input_text:
            self.status_label.configure(text="Errore: Inserisci i dati delle quote.", text_color="orange")
            self.results_display_textbox.configure(state="normal")
            self.results_display_textbox.delete("1.0", "end")
            self.results_display_textbox.insert("1.0", "Nessun input fornito.")
            self.results_display_textbox.configure(state="disabled")
            return

        lines = all_input_text.splitlines()
        output_results_list = []
        num_expected = len(self.expected_features)

        for i, line in enumerate(lines):
            line = line.strip()
            if not line:  # Salta righe vuote
                continue

            parsed_odds = self._parse_odds_line(line, num_expected)

            if parsed_odds is None:
                output_results_list.append(
                    f"Riga {i + 1} ('{line}'): Errore - formato input non valido o numero errato di quote (attese {num_expected}).")
                continue

            try:
                input_data_dict = dict(zip(self.expected_features, parsed_odds))
                input_df = pd.DataFrame([input_data_dict])  # DataFrame con una riga

                prediction_encoded_array = self.model.predict(input_df)
                prediction_encoded = prediction_encoded_array[0][0] if isinstance(prediction_encoded_array[0],
                                                                                  (list, np.ndarray)) else \
                prediction_encoded_array[0]

                probabilities_array = self.model.predict_proba(input_df)[0]

                predicted_display_label = PREDICTION_LABEL_MAP.get(prediction_encoded,
                                                                   f"Classe Sconosciuta ({prediction_encoded})")

                prob_texts_parts = []
                if self.model_classes_order_for_proba:  # Se abbiamo l'ordine delle classi
                    for idx, internal_class_code in enumerate(self.model_classes_order_for_proba):
                        display_label_for_prob = PREDICTION_LABEL_MAP.get(internal_class_code,
                                                                          f"Cl.{internal_class_code}")
                        short_display_label = display_label_for_prob.split(" ")[0]
                        prob_texts_parts.append(f"P({short_display_label}): {probabilities_array[idx]:.1%}")
                    probs_str = ", ".join(prob_texts_parts)
                else:  # Fallback se l'ordine delle classi non è noto
                    probs_str = "Probabilità non disponibili (ordine classi sconosciuto)"

                output_results_list.append(f"Input: [{line}] -> Risultato: {predicted_display_label} ({probs_str})")

            except Exception as e:
                output_results_list.append(
                    f"Riga {i + 1} ('{line}'): Errore durante la predizione - {str(e).splitlines()[0]}")
                print(f"Errore predizione per riga {i + 1} ('{line}'): {e}")

        self.results_display_textbox.configure(state="normal")
        self.results_display_textbox.delete("1.0", "end")
        if output_results_list:
            self.results_display_textbox.insert("1.0", "\n".join(output_results_list))
        else:
            self.results_display_textbox.insert("1.0", "Nessun input valido da processare.")
        self.results_display_textbox.configure(state="disabled")

        self.status_label.configure(text=f"Predizioni completate per {len(lines)} righe di input.",
                                    text_color="green" if any("->" in res for res in output_results_list) else "orange")


if __name__ == "__main__":
    if not os.path.exists(BASE_MODEL_PATH):
        print(f"ATTENZIONE: La cartella del modello '{BASE_MODEL_PATH}' non è stata trovata.")
    elif not os.path.exists(FULL_MODEL_PATH):
        print(f"ATTENZIONE: File modello '{MODEL_FILENAME}' non trovato in '{BASE_MODEL_PATH}'.")

    app = MulticlassPredictorApp()
    app.mainloop()
