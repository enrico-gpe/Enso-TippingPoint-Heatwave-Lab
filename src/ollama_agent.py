# ==============================================================================
# MODULE: src/ollama_agent.py
# PROJECT: EnsoHadley-Forecast-Lab
# Description: Agente semantico basato su Ollama (LLM locale).
#              Elabora le proiezioni numeriche Scikit-Learn e genera un report
#              bilingue (Italiano + Inglese) sulla dinamica rolling (4 settimane),
#              la fisica del Tipping Point dinamico (collasso gradiente termico /
#              rischio El Niño Permanente) e il forecast delle Heatwaves mediterranee.
# ==============================================================================

import requests
import json
import os
import sys
from datetime import datetime


def generate_clima_report(metrics_dict: dict, model_name: str = "llama3.2:3b", ollama_url: str = "http://localhost:11434") -> str:
    """
    Invia i dati numerici calcolati da Scikit-learn all'agente locale Ollama.
    Genera un report diagnostico bilingue (Italiano + Inglese) guidato da regole
    di inferenza dinamica basate sulla fisica del Tipping Point (gradiente Ovest-Est)
    e sulla previsione delle Onde di Calore (Heatwaves).
    """

    # --------------------------------------------------------------------------
    # PARSING & SANITIZZAZIONE DATI INPUT
    # --------------------------------------------------------------------------
    raw_attuale = metrics_dict.get('valore_attuale_obs', metrics_dict.get('valore_attuale_nino34', 1.988))
    raw_nino = metrics_dict.get('valore_target_nino34', metrics_dict.get('valore_luglio_nino34', 0.0))
    raw_hadley = metrics_dict.get('rischio_innesco_percent', 0.0)
    raw_sst_anom = metrics_dict.get('sst_global_anom', 1.35)
    raw_tipping = metrics_dict.get('tipping_point_nino', metrics_dict.get('tipping_point_nino34', 3.34))

    try:
        valore_attuale_float = float(raw_attuale)
        valore_nino_float = float(raw_nino)
        tipping_float = float(raw_tipping)
        raw_distanza = metrics_dict.get('distanza_da_tipping', metrics_dict.get('margine_da_tipping', valore_nino_float - tipping_float))
    except (ValueError, TypeError):
        valore_attuale_float = 1.988
        valore_nino_float = 0.0
        tipping_float = 3.34
        raw_distanza = 0.0

    regime_attivo = metrics_dict.get('regime_pliocenico_attivo', valore_nino_float >= tipping_float)

    valore_attuale_str = f"{valore_attuale_float:+.3f}"

    try:
        valore_nino_str = f"{valore_nino_float:+.3f}"
    except (ValueError, TypeError):
        valore_nino_str = f"+{raw_nino}" if not str(raw_nino).startswith('+') else str(raw_nino)

    try:
        rischio_hadley_float = float(raw_hadley)
        rischio_hadley_str = f"{rischio_hadley_float:.1f}"
    except (ValueError, TypeError):
        rischio_hadley_float = 0.0
        rischio_hadley_str = str(raw_hadley)

    try:
        tipping_point_str = f"{float(raw_tipping):+.2f}"
    except (ValueError, TypeError):
        tipping_point_str = str(raw_tipping)

    try:
        distanza_str = f"{float(raw_distanza):+.2f}"
    except (ValueError, TypeError):
        distanza_str = str(raw_distanza)

    data_esec = metrics_dict.get('execution_date', datetime.now().strftime("%d/%m/%Y"))
    settimane = metrics_dict.get('settimane_forecast', [])

    if settimane:
        finestra_str = f"dal {settimane[0]['start_date']} al {settimane[-1]['end_date']} (4 settimane)"
    else:
        finestra_str = "prossime 4 settimane"

    # Costruzione stringa di dettaglio del rischio settimanale e forecast heatwave
    dettaglio_settimanale_lines = []
    for s in settimane:
        label = s.get('label', 'Settimana').replace('\n', ' ')
        r_perc = s.get('rischio_percent', 'N/A')
        allerta = s.get('livello_allerta', 'NON DEFINITO')
        dettaglio_settimanale_lines.append(f"  - {label} ({s.get('start_date', '')} -> {s.get('end_date', '')}): Rischio {r_perc}% [{allerta}]")

    dettaglio_settimana_str = "\n".join(dettaglio_settimanale_lines) if dettaglio_settimanale_lines else "  - Proiezione settimanale in corso"

    # --------------------------------------------------------------------------
    # DINAMICA DEL PROMPT IN BASE ALLO STATO REALE (EVALUATION BRANCHING)
    # --------------------------------------------------------------------------
    if regime_attivo:
        status_tipping_rule = (
            f"SOGLIA SUPERATA: NINO3.4 ({valore_nino_str}°C) supera la soglia critica del Tipping Point ({tipping_point_str}°C). "
            f"Questo innesca l'annullamento del gradiente termico Pacifico Ovest-Est, l'arresto irreversibile del Feedback di Bjerknes "
            f"e dell'upwelling della Corrente di Humboldt, determinando la transizione a uno stato stazionario di El Niño Permanente (regime Pliocenico)."
        )
        livello_criticita_rule = "CRITICITÀ ROSSA / ESTREMA"
    else:
        status_tipping_rule = (
            f"SOGLIA NON SUPERATA: NINO3.4 ({valore_nino_str}°C) rimane INFERIORE al Tipping Point ({tipping_point_str}°C) "
            f"con un margine di sicurezza residuo di {distanza_str}°C. Il gradiente termico Pacifico Ovest-Est e l'upwelling di Humboldt rimangono funzionali. "
            f"Il sistema preserva la stabilità del regime Olocenico e la transizione a El Niño Permanente NON è attiva."
        )
        if rischio_hadley_float > 80.0:
            livello_criticita_rule = "CRITICITÀ ARANCIONE / ELEVATA (Saturazione Hadley Sostenuta & Rischio Heatwave)"
        elif rischio_hadley_float > 50.0:
            livello_criticita_rule = "CRITICITÀ GIALLA / MODERATA"
        else:
            livello_criticita_rule = "CRITICITÀ VERDE / ORDINARIA"

    # --------------------------------------------------------------------------
    # SYSTEM PROMPT & USER PROMPT BILINGUE CON RIGIDI VINCOLI ANTI-CONTRADDIZIONE
    # --------------------------------------------------------------------------
    system_instruction = (
        "Sei un climatologo teorico e geofisico di livello accademico. Genera SEMPRE "
        "il report richiesto in DUE LINGUE (Italiano seguito dalla traduzione in Inglese). "
        "Rispetti rigorosamente la struttura in 4 punti per entrambe le lingue. "
        "EVITA PAROLE INVENTATE COME 'OLOGRAFICO' O 'OLIGOCENICO': usa solo Olocene o Pliocene. "
        "Non contraddire mai lo stato del Tipping Point tra le varie sezioni."
    )

    user_prompt = f"""
Analizza la seguente diagnostica quantitativa generata dalla pipeline Scikit-learn per il run operativo del {data_esec} (Finestra: {finestra_str}):

--- PARAMETRI NUMERICI INPUT ---
- Anomalia SST NINO3.4 ISTANTANEA (CDAS Real-Time): {valore_attuale_str} °C
- Anomalia SST NINO3.4 PROIETTATA (Target a 4 Settimane): {valore_nino_str} °C
- Vettore Incremento Previsto (Δ-Trend): {valore_nino_float - valore_attuale_float:+.3f} °C
- Tipping Point Dinamico NINO3.4 (Collasso Gradiente Termico / El Niño Permanente): {tipping_point_str} °C
- Margine dal Tipping Point: {distanza_str} °C
- Probabilità saturazione Cella di Hadley: {rischio_hadley_str}%

--- DETTAGLIO PROIEZIONE SETTIMANALE HEATWAVE ED EVOLUZIONE MEDITERRANEA ---
{dettaglio_settimana_str}
--------------------------------

REGOLE TASSATIVE DI GENERAZIONE:
1. TARGET TEMPORALE: {valore_attuale_str}°C è l'osservazione attuale (CDAS), {valore_nino_str}°C è il target a 4 settimane.
2. SPIEGAZIONE TIPPING POINT (EL NIÑO PERMANENTE): Spiega che il Tipping Point ({tipping_point_str}°C) rappresenta la soglia di scomparsa del gradiente termico Pacifico Ovest-Est (Biforcazione di Bjerknes), con arresto dell'upwelling di Humboldt e rischio di transizione verso un El Niño Permanente (regime Pliocenico).
3. COERENZA TIPPING POINT: {status_tipping_rule}
4. FORECAST HEATWAVE MEDITERRANEE (PUNTO 3): Utilizza la progressione del rischio settimanale ({dettaglio_settimana_str}) per valutare direttamente l'intensificazione delle Onde di Calore (Heatwaves) e la formazione di strutture di blocco "Heat Dome" guidate dalla sussidenza della Cella di Hadley sul Mediterraneo. Poiché il margine dal Tipping Point è {distanza_str}°C (sotto soglia), NON ipotizzare un collasso Pliocenico permanente nel punto 3.
5. LIVELLO ALLERTA ESATTO NEL PUNTO 4: Dichiara espressamente **{livello_criticita_rule}**.

FORMATO OUTPUT RICHIESTO:

### SECTION 1: RAPPORTO DIAGNOSTICO (ITALIANO)
1. **Accoppiamento Oceanico e Tipping Point (Rischio El Niño Permanente):** 
   Confronta l'anomalia attuale ({valore_attuale_str} °C) con la proiezione ({valore_nino_str} °C). Spiega il Tipping Point ({tipping_point_str} °C) come perdita del gradiente termico Pacifico Ovest-Est e potenziale transizione a El Niño permanente, evidenziando come il margine residuo ({distanza_str} °C) garantisca la stabilità Olocenica.
2. **Impatto sulla Cella di Hadley e Sussidenza:** 
   Analizza la risposta della sussidenza atmosferica al target proiettato ({rischio_hadley_str}%) e la saturazione della cella meridionale.
3. **Forecast Heatwave e Teleconnessione Mediterranea:** 
   Analizza l'evoluzione settimanale del rischio ({dettaglio_settimana_str}) prevedendo la probabilità di Onde di Calore (Heatwaves) e la persistenza di strutture a cupola di calore (Heat Dome) sul Mediterraneo.
4. **Livello di Criticità Operativa:** 
   Dichiara espressamente **{livello_criticita_rule}**.

---

### SECTION 2: DIAGNOSTIC REPORT (ENGLISH TRANSLATION)
1. **Oceanic Coupling & Permanent El Niño Tipping Point:** 
   Compare current anomaly ({valore_attuale_str} °C) with projection ({valore_nino_str} °C). Explain the Tipping Point ({tipping_point_str} °C) as the collapse of the West-East Pacific thermal gradient and permanent El Niño state risk, noting the safety margin ({distanza_str} °C) preserving Holocene stability.
2. **Impact on Hadley Cell:** 
   Analyse atmospheric subsidence response to the projected target ({rischio_hadley_str}%) and tropical circulation saturation.
3. **Heatwave Forecast and Mediterranean Teleconnection:** 
   Detail the weekly breakdown ({dettaglio_settimana_str}) evaluating Mediterranean Heatwave risks and Subtropical High / Heat Dome persistence.
4. **Operational Criticality Level:** 
   State explicitly **{livello_criticita_rule}** (translated).
"""

    payload = {
        "model": model_name,
        "system": system_instruction,
        "prompt": user_prompt,
        "stream": False,
        "options": {
            "temperature": 0.1,  # Temperatura ridotta per massima stabilità concettuale e aderenza scientifica
            "top_p": 0.85
        }
    }

    endpoint = f"{ollama_url}/api/generate"

    try:
        response = requests.post(endpoint, json=payload, timeout=90)
        response.raise_for_status()
        result_json = response.json()
        report_text = result_json.get("response", "[ERRORE] Risposta vuota ricevuta da Ollama.")

        # Salvataggio automatico del report bilingue
        os.makedirs("outputs", exist_ok=True)
        with open("outputs/report_rolling_4_settimane.md", "w", encoding="utf-8") as f:
            f.write(report_text)

        return report_text

    except requests.exceptions.ConnectionError:
        return (
            f"[ERRORE DI CONNESSIONE] Impossibile raggiungere Ollama su {ollama_url}.\n"
            "Verifica che il servizio sia attivo eseguendo: `ollama serve`"
        )
    except requests.exceptions.Timeout:
        return "[TIMEOUT] La richiesta ad Ollama ha impiegato troppo tempo."
    except Exception as e:
        return f"[ERRORE GENERICO] Eccezione durante la chiamata ad Ollama: {str(e)}"


if __name__ == "__main__":
    try:
        from ml_engine import run_rolling_dms_forecast
    except ImportError:
        from src.ml_engine import run_rolling_dms_forecast

    print("=== ESECUZIONE RUN MODEL REAL-TIME PER TEST OLLAMA AGENT ===")
    test_data = run_rolling_dms_forecast()

    print(f"Data Run: {test_data['execution_date']}")
    print(f"Valore Attuale CDAS: +{test_data['valore_attuale_obs']:.3f} °C")
    print(f"Target Proiettato 4-Sett: +{test_data['valore_target_nino34']:.3f} °C")
    print(f"Tipping Point: +{test_data['tipping_point_nino']:.2f} °C")
    print("-" * 60)
    print("=== GENERAZIONE REPORT SEMANTICO BILINGUE OLLAMA ===")

    report = generate_clima_report(test_data, model_name="llama3.2:3b")
    print(report)