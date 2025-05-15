import customtkinter
import pandas as pd
from catboost import CatBoostClassifier
import os
import re # Importato per la gestione dell'input multi-quota

# --- Costanti ---
MODEL_FILENAME = "catboost_draw_predictor_production.cbm"
FEATURES_TO_USE = ['odds_1']  # Feature usata dal modello


class DrawPredictorApp(customtkinter.CTk):
    def __init__(self):
        super().__init__()

        self.title("Predittore Pareggi Calcio (Multi-Quota)")
        self.geometry("550x600")  # Aumentate le dimensioni per ospitare più output
        customtkinter.set_appearance_mode("System")  # "Dark", "Light", "System"
        customtkinter.set_default_color_theme("blue")  # Tema di colore

        self.model = None
        self.model_loaded_successfully = False

        # Caricamento del modello
        self._load_model()

        # Creazione dei widget dell'interfaccia utente
        self._create_widgets()

    def _load_model(self):
        """Carica il modello CatBoost salvato."""
        if not os.path.exists(MODEL_FILENAME):
            print(f"ERRORE: File modello '{MODEL_FILENAME}' non trovato.")
            # Non possiamo procedere senza modello, lo segnaleremo nell'UI.
            return

        try:
            self.model = CatBoostClassifier()
            self.model.load_model(MODEL_FILENAME)
            self.model_loaded_successfully = True
            print(f"Modello '{MODEL_FILENAME}' caricato con successo.")
        except Exception as e:
            print(f"ERRORE durante il caricamento del modello '{MODEL_FILENAME}': {e}")
            self.model = None
            self.model_loaded_successfully = False # Assicura che sia False

    def _create_widgets(self):
        """Crea e posiziona i widget nell'interfaccia."""

        main_frame = customtkinter.CTkFrame(self)
        main_frame.pack(pady=20, padx=20, fill="both", expand=True)

        app_title_label = customtkinter.CTkLabel(main_frame, text="Predittore Pareggi Multi-Quota",
                                                 font=customtkinter.CTkFont(size=20, weight="bold"))
        app_title_label.pack(pady=(10, 20))

        # Etichetta e campo di input per le quote (modificato per multi-input)
        odds_label = customtkinter.CTkLabel(main_frame, text="Inserisci le quote 'Odds 1' (separate da virgola o una per riga):")
        odds_label.pack(pady=(10, 5))

        self.odds_textbox = customtkinter.CTkTextbox(main_frame, height=100, width=300) # Aumentata larghezza
        self.odds_textbox.pack(pady=5)
        self.odds_textbox.insert("1.0", "Esempio:\n2.5\n3.1, 4.0\n1.95") # Testo placeholder

        # Pulsante per avviare la predizione
        self.predict_button = customtkinter.CTkButton(main_frame, text="Predici Risultati",
                                                      command=self._on_predict_button_click)
        self.predict_button.pack(pady=20)

        if not self.model_loaded_successfully:
            self.predict_button.configure(state="disabled")

        # Etichetta per i risultati
        result_intro_label = customtkinter.CTkLabel(main_frame, text="Risultati Predetti:",
                                                         font=customtkinter.CTkFont(size=14, weight="bold"))
        result_intro_label.pack(pady=(10,0))

        # Area di testo per mostrare i risultati multipli
        self.results_display_textbox = customtkinter.CTkTextbox(main_frame, height=200, width=450, state="disabled") # state="disabled" per renderlo read-only
        self.results_display_textbox.pack(pady=10, fill="x", expand=True)


        # Etichetta di stato per caricamento modello o errori
        self.status_label = customtkinter.CTkLabel(self, text="", font=customtkinter.CTkFont(size=10))
        self.status_label.pack(side="bottom", pady=5, fill="x")

        if not self.model_loaded_successfully:
            self.status_label.configure(text=f"Errore: Modello '{MODEL_FILENAME}' non trovato o corrotto. Impossibile predire.",
                                        text_color="red")
            self.results_display_textbox.configure(state="normal")
            self.results_display_textbox.delete("1.0", "end")
            self.results_display_textbox.insert("1.0", "Impossibile predire: modello non caricato.")
            self.results_display_textbox.configure(state="disabled")
        else:
            self.status_label.configure(text=f"Modello '{MODEL_FILENAME}' caricato.", text_color="green")
            self.results_display_textbox.configure(state="normal")
            self.results_display_textbox.delete("1.0", "end")
            self.results_display_textbox.insert("1.0", "- In attesa di input -")
            self.results_display_textbox.configure(state="disabled")

    def _parse_odds_input(self, odds_input_str: str) -> list[str]:
        """
        Analizza la stringa di input per estrarre le quote.
        Le quote possono essere separate da virgole, spazi o newline.
        """
        # Sostituisce virgole e newline con uno spazio, poi splitta per spazi
        # Rimuove stringhe vuote risultanti da spazi multipli
        odds_list = re.split(r'[,\s\n]+', odds_input_str)
        return [odd.strip() for odd in odds_list if odd.strip()]


    def _on_predict_button_click(self):
        """Gestisce l'evento click del pulsante di predizione per una lista di quote."""
        if not self.model_loaded_successfully or self.model is None:
            self.results_display_textbox.configure(state="normal")
            self.results_display_textbox.delete("1.0", "end")
            self.results_display_textbox.insert("1.0", "Errore: Modello non caricato.")
            self.results_display_textbox.configure(state="disabled", text_color="red")
            self.status_label.configure(text="Errore: Modello non caricato.", text_color="red")
            return

        odds_input_str = self.odds_textbox.get("1.0", "end-1c") # Legge tutto il testo
        if not odds_input_str.strip() or odds_input_str.strip() == "Esempio:\n2.5\n3.1, 4.0\n1.95": # Controllo se l'input è vuoto o solo placeholder
            self.results_display_textbox.configure(state="normal")
            self.results_display_textbox.delete("1.0", "end")
            self.results_display_textbox.insert("1.0", "Errore: Inserisci almeno una quota.")
            self.results_display_textbox.configure(state="disabled", text_color="orange")
            self.status_label.configure(text="Errore: Input quote mancante.", text_color="orange")
            return

        odds_str_list = self._parse_odds_input(odds_input_str)

        if not odds_str_list:
            self.results_display_textbox.configure(state="normal")
            self.results_display_textbox.delete("1.0", "end")
            self.results_display_textbox.insert("1.0", "Errore: Nessuna quota valida trovata nell'input.")
            self.results_display_textbox.configure(state="disabled", text_color="orange")
            self.status_label.configure(text="Errore: Input quote non valido.", text_color="orange")
            return

        results_output = [] # Lista per accumulare le stringhe di risultato

        for i, odd_str in enumerate(odds_str_list):
            try:
                odds_value = float(odd_str)
            except ValueError:
                results_output.append(f"Quota '{odd_str}': Errore - Valore non numerico.")
                continue # Passa alla prossima quota

            # Prepara i dati per il modello
            input_data = pd.DataFrame({FEATURES_TO_USE[0]: [odds_value]})

            try:
                # Effettua la predizione
                prediction = self.model.predict(input_data)
                probability = self.model.predict_proba(input_data)

                predicted_class = prediction[0]
                probability_draw = probability[0][1] # Probabilità della classe '1' (Pareggio)

                result_text = "PAREGGIO" if predicted_class == 1 else "NON PAREGGIO"
                result_color = "green" if predicted_class == 1 else "red" # Teoricamente non possiamo colorare singole linee nel textbox standard

                results_output.append(f"Quota {odds_value:.2f}: {result_text} (Prob. Pareggio: {probability_draw:.2%})")

            except Exception as e:
                error_detail = str(e).split('\n')[0] # Prende solo la prima riga dell'errore
                results_output.append(f"Quota {odds_value:.2f}: Errore durante la predizione. ({error_detail[:50]}...)")
                print(f"Errore di predizione per quota {odds_value}: {e}")

        # Aggiorna il textbox dei risultati
        self.results_display_textbox.configure(state="normal") # Abilita la modifica
        self.results_display_textbox.delete("1.0", "end")      # Cancella il contenuto precedente
        self.results_display_textbox.insert("1.0", "\n".join(results_output)) # Inserisce i nuovi risultati
        self.results_display_textbox.configure(state="disabled") # Disabilita la modifica (read-only)

        self.status_label.configure(text=f"Predizioni completate per {len(odds_str_list)} quote.", text_color="gray")


if __name__ == "__main__":
    # Assicurati che il file modello esista o crea un dummy per testare l'UI
    # Questo è solo per scopi di test se non hai il file .cbm
    if not os.path.exists(MODEL_FILENAME):
        print(f"ATTENZIONE: File modello '{MODEL_FILENAME}' non trovato. L'applicazione funzionerà in modalità limitata.")
        # Potresti voler creare un modello fittizio per testare l'interfaccia
        # from catboost import CatBoostClassifier
        # dummy_model = CatBoostClassifier(iterations=1, verbose=0)
        # dummy_model.fit([[0]], [0]) # Addestramento fittizio
        # dummy_model.save_model(MODEL_FILENAME)
        # print(f"Creato un modello fittizio '{MODEL_FILENAME}' per test.")
        # In questo caso, lasciamo che l'app segnali l'errore di caricamento.

    app = DrawPredictorApp()
    app.mainloop()
