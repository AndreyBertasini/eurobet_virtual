import requests
import pandas as pd
from datetime import datetime, timedelta
import time
import json
import os
import gzip  # Importato per la decompressione manuale gzip
import brotli  # Importato per la decompressione manuale Brotli


class VirtualSportsCollector:
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/136.0.0.0 Safari/537.36',
            'Accept': 'application/json, text/plain, */*',
            # Indichiamo che accettiamo risposte compresse (gzip, deflate, br)
            'Accept-Encoding': 'gzip, deflate, br',
            'Accept-Language': 'en-US,en;q=0.6',
            'Origin': 'https://www.eurobet.it',
            'Referer': 'https://www.eurobet.it/',
            'X-EB-Accept-Language': 'it_IT',
            'X-EB-MarketId': '5',
            'X-EB-PlatformId': '1',
            'Connection': 'keep-alive',
            'Sec-Fetch-Dest': 'empty',
            'Sec-Fetch-Mode': 'cors',
            'Sec-Fetch-Site': 'same-site'
        }
        self.base_url = "https://virtualservice.eurobet.it/virtual-winning-service/virtual-schedule/services/winningresult/68/17/{}"
        self.csv_filename = "virtual_matches_data.csv"
        self.excel_filename = "virtual_matches_data.xlsx"

    def create_match_id(self, row):
        """Crea un identificatore univoco per ogni partita."""
        # Assicura che i campi chiave siano stringhe per evitare errori con None
        date_val = str(row.get('date', ''))
        hour_val = str(row.get('hour', ''))
        home_team_val = str(row.get('home_team', ''))
        away_team_val = str(row.get('away_team', ''))
        return f"{date_val}_{hour_val}_{home_team_val}_{away_team_val}"

    def load_existing_data(self):
        """Carica i dati esistenti dal CSV, se esiste."""
        if os.path.exists(self.csv_filename):
            try:
                # Specifica dtype per colonne potenzialmente problematiche per evitare avvisi di tipo misto
                # Questo è un esempio, potrebbe essere necessario adattarlo in base ai dati effettivi
                dtype_spec = {
                    'odds_1': 'object',
                    'result': 'object',
                    'over_under_25': 'object',
                    'odds_over_under_25': 'object',
                    'goal_no_goal': 'object',
                    'odds_goal_no_goal': 'object'
                }
                df = pd.read_csv(self.csv_filename, dtype=dtype_spec)
                # Converte la colonna datetime se esiste, gestendo errori
                if 'datetime' in df.columns:
                    df['datetime'] = pd.to_datetime(df['datetime'], errors='coerce')
                return df
            except pd.errors.EmptyDataError:
                print(f"Il file CSV {self.csv_filename} è vuoto. Verrà creato un nuovo DataFrame.")
                return pd.DataFrame()
            except Exception as e:
                print(f"Errore durante il caricamento del file CSV {self.csv_filename}: {e}")
                return pd.DataFrame()
        return pd.DataFrame()

    def get_virtual_data(self, start_date, end_date):
        """
        Recupera i dati virtuali per l'intervallo di date specificato.
        Args:
            start_date (datetime): La data di inizio per il recupero dei dati.
            end_date (datetime): La data di fine (inclusa) per il recupero dei dati.
        """
        all_matches = []
        current_date = start_date

        while current_date <= end_date:
            date_str = current_date.strftime("%d-%m-%Y")
            url = self.base_url.format(date_str)
            print(f"Tentativo di recupero dati per {date_str} da URL: {url}")
            data = None
            response = None  # Inizializza response

            try:
                response = requests.get(url, headers=self.headers, timeout=20)  # Timeout aumentato leggermente
                response.raise_for_status()  # Solleva eccezione per status code 4xx/5xx

                if not response.content:
                    print(f"Risposta vuota ricevuta per {date_str} (Status: {response.status_code}). URL: {url}")
                    time.sleep(1)
                    current_date += timedelta(days=1)
                    continue

                # Tentativo di parsing JSON (requests dovrebbe gestire la decompressione automatica)
                try:
                    data = response.json()
                    print(f"Dati per {date_str} parsati con successo (decompressione automatica o nessuna).")
                except json.JSONDecodeError:
                    print(f"Errore di decodifica JSON standard per {date_str}.")
                    print(f"Status code della risposta: {response.status_code}")
                    content_encoding = response.headers.get('Content-Encoding', '').lower()
                    print(f"Header Content-Encoding: {content_encoding if content_encoding else 'Non presente'}")

                    if content_encoding == 'br':
                        print("Il contenuto sembra essere Brotli. Tentativo di decomprimere manualmente.")
                        try:
                            decompressed_content = brotli.decompress(response.content)
                            data = json.loads(decompressed_content.decode('utf-8'))
                            print(f"Contenuto Brotli per {date_str} decompresso e parsato con successo manualmente.")
                        except Exception as e_decompress:
                            print(
                                f"Fallimento nella decompressione manuale Brotli o nel parsing per {date_str}: {e_decompress}")
                            print(f"Contenuto grezzo (primi 200 byte): {response.content[:200]}...")

                    elif content_encoding == 'gzip' or response.content.startswith(b'\x1f\x8b\x08'):
                        print("Il contenuto sembra essere Gzip. Tentativo di decomprimere manualmente.")
                        try:
                            decompressed_content = gzip.decompress(response.content)
                            data = json.loads(decompressed_content.decode('utf-8'))
                            print(f"Contenuto Gzip per {date_str} decompresso e parsato con successo manualmente.")
                        except Exception as e_decompress:
                            print(
                                f"Fallimento nella decompressione manuale Gzip o nel parsing per {date_str}: {e_decompress}")
                            print(f"Contenuto grezzo (primi 200 byte): {response.content[:200]}...")
                    else:
                        print(
                            f"Il contenuto non sembra essere Brotli o Gzip, o la decodifica JSON è fallita per altri motivi.")
                        print(
                            f"Testo della risposta (se decodificabile, primi 500 caratteri): {response.text[:500] if response.text else 'Nessun testo decodificabile'}...")
                        print(f"Contenuto grezzo (primi 200 byte): {response.content[:200]}...")

                    if data is None:  # Se tutti i tentativi di decodifica falliscono
                        time.sleep(1)
                        current_date += timedelta(days=1)
                        continue  # Passa alla data successiva

                # Elaborazione dei dati se il parsing JSON ha avuto successo
                if data and 'result' in data and data['result'] is not None and 'groupDate' in data['result']:
                    for group in data['result']['groupDate']:
                        if 'events' in group and group['events'] is not None:
                            for event in group['events']:
                                try:
                                    home_team_name = event.get('eventDescription', ' - ').split(' - ')[0]
                                    away_team_name = event.get('eventDescription', ' - ')[1] if ' - ' in event.get(
                                        'eventDescription', '') else None

                                    match_data = {
                                        'date': event.get('date'),
                                        'hour': event.get('hour'),
                                        'home_team': home_team_name,
                                        'away_team': away_team_name,
                                        'score': event.get('finalResult'),
                                        'home_goals': None,
                                        'away_goals': None,
                                        'datetime': None
                                    }

                                    if event.get('date') and event.get('hour'):
                                        try:
                                            match_data['datetime'] = pd.to_datetime(f"{event['date']} {event['hour']}",
                                                                                    format='%d-%m-%Y %H:%M:%S',
                                                                                    errors='coerce')
                                        except ValueError as ve:
                                            print(
                                                f"Errore di formato data/ora per {event['date']} {event['hour']}: {ve}")
                                            match_data['datetime'] = None  # Assicura che sia None in caso di errore

                                    if event.get('finalResult') and '-' in event['finalResult']:
                                        try:
                                            scores = event['finalResult'].split('-')
                                            match_data['home_goals'] = int(scores[0])
                                            match_data['away_goals'] = int(scores[1])
                                        except ValueError:
                                            print(
                                                f"Impossibile parsare il punteggio: {event['finalResult']} per l'evento {event.get('eventDescription')}")

                                    if 'oddGroup' in event and event['oddGroup'] is not None:
                                        for odd_group in event['oddGroup']:
                                            bet_abbr = odd_group.get('betDescriptionAbbr')
                                            odds_list = odd_group.get('odds')
                                            result_desc_list = odd_group.get('resultDescription')

                                            # Controllo che odds_list e result_desc_list non siano None prima di accedere agli elementi
                                            if bet_abbr == '1X2' and odds_list and result_desc_list:
                                                match_data['odds_1'] = odds_list[0] if len(odds_list) > 0 else None
                                                match_data['result'] = result_desc_list[0] if len(
                                                    result_desc_list) > 0 else None
                                            elif bet_abbr == 'U/O 2.5' and odds_list and result_desc_list:  # Corretto da 'Under / Over 2.5'
                                                match_data['over_under_25'] = result_desc_list[0] if len(
                                                    result_desc_list) > 0 else None
                                                match_data['odds_over_under_25'] = odds_list[0] if len(
                                                    odds_list) > 0 else None
                                            elif bet_abbr == 'Goal/No Goal' and odds_list and result_desc_list:
                                                match_data['goal_no_goal'] = result_desc_list[0] if len(
                                                    result_desc_list) > 0 else None
                                                match_data['odds_goal_no_goal'] = odds_list[0] if len(
                                                    odds_list) > 0 else None

                                    all_matches.append(match_data)
                                except Exception as e_inner:
                                    print(
                                        f"Errore durante l'elaborazione di un evento per {date_str}: {e_inner}. Evento: {json.dumps(event, indent=2)}")
                        else:
                            print(f"Nessun evento ('events') trovato nel gruppo per {date_str} o il gruppo è None.")
                elif data:
                    print(
                        f"Struttura JSON inattesa o 'result'/'groupDate' mancante per {date_str}. Dati ricevuti (primi 500 caratteri): {str(data)[:500]}...")
                else:  # data is None
                    print(f"Nessun dato JSON valido ottenuto per {date_str} dopo i tentativi di decompressione.")


            except requests.exceptions.HTTPError as http_err:
                print(
                    f"Errore HTTP per {date_str}: {http_err} (Status: {response.status_code if response else 'N/A'}) - URL: {url}")
            except requests.exceptions.ConnectionError as conn_err:
                print(f"Errore di connessione per {date_str}: {conn_err} - URL: {url}")
            except requests.exceptions.Timeout as timeout_err:
                print(f"Errore di Timeout per {date_str}: {timeout_err} - URL: {url}")
            except Exception as e:
                print(f"Errore imprevisto durante il recupero dati per {date_str}: {e} - URL: {url}")

            time.sleep(1.5)  # Aumentata leggermente la pausa
            current_date += timedelta(days=1)

        return pd.DataFrame(all_matches)

    def merge_and_save_data(self, new_data):
        """Unisci i nuovi dati con quelli esistenti, rimuovi i duplicati e salva."""
        existing_data = self.load_existing_data()

        if new_data.empty:
            print("Nessun nuovo dato da unire.")
            if not existing_data.empty:
                print(f"Database esistente contiene {len(existing_data)} partite.")
                try:
                    existing_data.to_csv(self.csv_filename, index=False)
                    existing_data.to_excel(self.excel_filename, index=False)
                    print(f"Dati esistenti (ri)salvati in {self.csv_filename} e {self.excel_filename}")
                except Exception as e:
                    print(f"Errore durante il salvataggio dei dati esistenti: {e}")
            else:
                print("Database esistente è vuoto o non è stato possibile caricarlo.")
            return existing_data

        # Assicura che la colonna 'datetime' sia nel formato corretto prima di unire
        if not existing_data.empty and 'datetime' in existing_data.columns:
            existing_data['datetime'] = pd.to_datetime(existing_data['datetime'], errors='coerce')

        if 'datetime' in new_data.columns:
            new_data['datetime'] = pd.to_datetime(new_data['datetime'], errors='coerce')
        else:
            print(
                "Attenzione: colonna 'datetime' mancante nei nuovi dati. Impossibile ordinare o rimuovere duplicati basati su di essa.")
            # Potrebbe essere necessario creare una colonna datetime fittizia o gestire l'errore
            # Per ora, procediamo ma l'ordinamento e la rimozione dei duplicati potrebbero non funzionare come previsto.

        combined_data = pd.concat([existing_data, new_data], ignore_index=True)

        # Rimuovi righe dove 'datetime' è NaT (Not a Time) solo se la colonna esiste
        if 'datetime' in combined_data.columns:
            combined_data.dropna(subset=['datetime'], inplace=True)

        # Crea 'match_id' per la deduplicazione
        # Usa .get() con valori predefiniti vuoti per evitare errori se le chiavi mancano
        combined_data['match_id'] = combined_data.apply(
            lambda
                row: f"{row.get('date', '')}_{row.get('hour', '')}_{row.get('home_team', '')}_{row.get('away_team', '')}",
            axis=1
        )

        # Rimuovi righe dove match_id è vuoto o indica dati incompleti
        combined_data = combined_data[combined_data['match_id'] != "___"]
        combined_data.dropna(subset=['match_id'], inplace=True)  # Rimuove eventuali None rimasti

        # Ordinamento e rimozione duplicati
        if 'datetime' in combined_data.columns:
            combined_data = combined_data.sort_values('datetime', ascending=False)
            combined_data = combined_data.drop_duplicates(subset=['match_id'], keep='first')
        else:
            # Se datetime non è disponibile, deduplica solo su match_id senza un ordinamento temporale specifico
            combined_data = combined_data.drop_duplicates(subset=['match_id'], keep='first')

        if 'match_id' in combined_data.columns:
            combined_data = combined_data.drop('match_id', axis=1)

        try:
            combined_data.to_csv(self.csv_filename, index=False)
            combined_data.to_excel(self.excel_filename, index=False)
            print(f"Dati combinati salvati con successo in {self.csv_filename} e {self.excel_filename}")
        except Exception as e:
            print(f"Errore durante il salvataggio dei file combinati: {e}")

        return combined_data

    def collect_data(self, days_to_fetch_count=1):
        """
        Metodo principale per raccogliere ed elaborare i dati.
        Recupera i dati per `days_to_fetch_count` giorni, terminando con ieri.
        """
        if not isinstance(days_to_fetch_count, int) or days_to_fetch_count < 1:
            print("days_to_fetch_count deve essere un intero positivo.")
            return

        # La data di oggi
        today = datetime.now()
        # La data finale per la raccolta è ieri
        end_date_of_collection = today - timedelta(days=1)
        # La data iniziale per la raccolta
        start_date_of_collection = end_date_of_collection - timedelta(days=days_to_fetch_count - 1)

        print(
            f"Richiesta dati dal {start_date_of_collection.strftime('%d-%m-%Y')} al {end_date_of_collection.strftime('%d-%m-%Y')}")

        new_data = self.get_virtual_data(start_date_of_collection, end_date_of_collection)

        if not new_data.empty:
            print(f"Recuperate {len(new_data)} nuove partite.")
        else:
            print("Nessun nuovo dato raccolto dalla fonte.")

        final_data = self.merge_and_save_data(new_data)  # Chiama sempre merge_and_save_data

        if final_data is not None and not final_data.empty:
            print(f"Totale partite nel database dopo l'aggiornamento: {len(final_data)}")
        elif final_data is not None and final_data.empty and new_data.empty:
            print("Il database è vuoto e non sono stati aggiunti nuovi dati.")
        elif final_data is not None and final_data.empty and not new_data.empty:
            print("Errore: nuovi dati recuperati ma il database finale è vuoto dopo l'unione.")


def main(days_to_fetch_count=1):
    """
    Funzione principale per avviare il collettore di dati.
    Args:
        days_to_fetch_count (int): Numero di giorni passati per cui recuperare i dati.
                                   Default è 1 (solo i dati di ieri).
    """
    collector = VirtualSportsCollector()
    collector.collect_data(days_to_fetch_count)


if __name__ == "__main__":
    # --- IMPOSTAZIONE DEI GIORNI DA RECUPERARE ---
    # Modifica questa variabile per cambiare il numero di giorni passati da cui recuperare i dati.
    # Esempio: 1 recupera solo i dati di ieri.
    # Esempio: 7 recupera i dati degli ultimi 7 giorni (fino a ieri).
    GIORNI_DA_RECUPERARE = 3  # Modificato per testare su un range più ampio come da log

    print(f"Avvio dello script per recuperare i dati degli ultimi {GIORNI_DA_RECUPERARE} giorni (fino a ieri).")
    main(days_to_fetch_count=GIORNI_DA_RECUPERARE)
    print("Script terminato.")
