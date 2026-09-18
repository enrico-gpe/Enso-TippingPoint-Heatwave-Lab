# ==============================================================================
# SCRIPT: main.py
# PROJECT: EnsoHadley-Forecast-Lab
# Description: Entry point principale. Esegue la pipeline end-to-end:
#              Data Pipeline (Scikit-Learn DMS-MLP + Tipping Point Dinamico) -> 
#              Visual Dashboard (Matplotlib/Seaborn) con Auto-Plot & Preview -> 
#              Reasoning & Report Generation (Ollama LLM - llama3.2:3b).
# ==============================================================================

# ----------------------------------------------------------------------

# Copyright (C) 2026 Enrico Pozzi
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.
# ----------------------------------------------------------------------


import os
import sys
import platform
import subprocess
import webbrowser

# Importazione dei moduli interni
from src.ml_engine import run_rolling_dms_forecast
from src.plotter import generate_diagnostic_dashboard
from src.ollama_agent import generate_clima_report


def open_generated_image(image_path: str):
    """
    Apre automaticamente l'immagine salvata utilizzando il visualizzatore
    predefinito del sistema operativo (macOS, Windows, Linux).
    """
    if not os.path.exists(image_path):
        return

    try:
        current_os = platform.system()
        if current_os == 'Darwin':         # macOS
            subprocess.run(['open', image_path], check=False)
        elif current_os == 'Windows':      # Windows
            os.startfile(image_path)
        elif current_os == 'Linux':        # Linux
            subprocess.run(['xdg-open', image_path], check=False)
        else:                              # Fallback universale via browser
            webbrowser.open(os.path.abspath(image_path))
    except Exception as e:
        print(f"      ⚠️  Impossibile aprire l'immagine automaticamente: {e}")


def main():
    print("\n" + "=" * 75)
    print("   ENSO-HADLEY FORECAST LAB — RUN OPERATIVO ROLLING (4 SETTIMANE)")
    print("=" * 75 + "\n")
    
    # Assicura l'esistenza delle directory necessarie
    os.makedirs("outputs", exist_ok=True)
    os.makedirs("models", exist_ok=True)
    
    # Configurazione Modello Ollama
    MODEL_OLLAMA = "llama3.2:3b"
    
    # --------------------------------------------------------------------------
    # FASE 1: EXECUTION ENGINE NUMERICO (Scikit-Learn Rolling DMS-MLP)
    # --------------------------------------------------------------------------
    print("[1/3] Scaricamento dati NOAA ed esecuzione modello Rolling DMS-MLP...")
    try:
        results = run_rolling_dms_forecast(lags=12, n_ahead=12, n_members=30)
        
        # Estrattori con fallback per garantire massima compatibilità
        data_esec = results.get("execution_date", "Oggi")
        valore_nino = float(results.get("valore_target_nino34", results.get("valore_luglio_nino34", 0.0)))
        tipping_point = float(results.get("tipping_point_nino", results.get("tipping_point_nino34", 2.23)))
        distanza_tipping = float(results.get("distanza_da_tipping", results.get("margine_da_tipping", valore_nino - tipping_point)))
        regime_attivo = results.get("regime_pliocenico_attivo", valore_nino >= tipping_point)
        rischio_hadley = float(results.get("rischio_innesco_percent", 0.0))
        settimane = results.get("settimane_forecast", [])
        
        print(f"      ✔ Run eseguito in data:                 {data_esec}")
        print(f"      ✔ Target Proiettato NINO3.4:          +{valore_nino:.3f} °C")
        print(f"      ✔ Tipping Point Dinamico (Bjerknes): +{tipping_point:.2f} °C")
        print(f"      ✔ Margine / Scostamento dal Tipping: {distanza_tipping:+.2f} °C")
        print(f"      ✔ Regime Pliocenico Attivo:          {regime_attivo}")
        print(f"      ✔ Rischio Innesco Saturazione Hadley: {rischio_hadley:.1f}%\n")
        
        if settimane:
            print("      📅 Finestra Temporale Mobile (4 Settimane):")
            for s in settimane:
                label_pulita = s['label'].replace('\n', ' ')
                print(f"         • {label_pulita}: {s.get('start_date', 'N/D')} ➔ {s.get('end_date', 'N/D')}")
            
    except Exception as e:
        print(f"      ❌ [ERRORE CRITICO] Fallimento nell'engine ML: {e}")
        sys.exit(1)
        
    # --------------------------------------------------------------------------
    # FASE 2: GENERAZIONE E AUTO-PLOT GRAPHIC DASHBOARD
    # --------------------------------------------------------------------------
    print("\n[2/3] Generazione e rendering cruscotto grafico ad alta risoluzione...")
    output_image_path = "outputs/heatwave_diagnostic_rolling.png"
    try:
        # Genera il grafico e richiede la visualizzazione (show_plot=True se supportato)
        generate_diagnostic_dashboard(results, output_path=output_image_path, show_plot=True)
        print(f"      ✔ Grafico salvato in: {output_image_path}")
        
        # Apertura automatica del file PNG nel visualizzatore di sistema
        open_generated_image(output_image_path)
        print("      🚀 Output grafico aperto automaticamente a schermo!")

    except TypeError:
        # Fallback se generate_diagnostic_dashboard non accetta show_plot
        generate_diagnostic_dashboard(results, output_path=output_image_path)
        print(f"      ✔ Grafico salvato in: {output_image_path}")
        open_generated_image(output_image_path)
        print("      🚀 Output grafico aperto automaticamente a schermo!")
    except Exception as e:
        print(f"      ❌ [ERRORE] Fallimento nella generazione grafica: {e}")

    # --------------------------------------------------------------------------
    # FASE 3: REASONING PALEOCLIMATICO E REPORT DIAGNOSTICO (Ollama LLM)
    # --------------------------------------------------------------------------
    print(f"\n[3/3] Connessione all'agente locale Ollama (`{MODEL_OLLAMA}`)...")
    
    report_text = generate_clima_report(results, model_name=MODEL_OLLAMA)
    
    # --------------------------------------------------------------------------
    # SALVATAGGIO REPORT FINALE IN MARKDOWN
    # --------------------------------------------------------------------------
    report_md_path = "outputs/report_rolling_4_settimane.md"
    
    if settimane:
        elenco_settimane_md = "\n".join([
            f"* **{s['label'].replace('\n', ' ')}:** dal `{s.get('start_date', 'N/D')}` al `{s.get('end_date', 'N/D')}`"
            for s in settimane
        ])
    else:
        elenco_settimane_md = "* *Finestra mobile standard a 4 settimane.*"
    
    report_header = f"""# BOLLETTINO DIAGNOSTICO METEOCLIMATICO – ROLLING 4 SETTIMANE
**Data di Esecuzione Run:** {data_esec}  
**Engine Numerico:** Scikit-Learn Pipeline (`MLPRegressor` Rolling DMS)  
**Agente Diagnostico:** Ollama LLM (`{MODEL_OLLAMA}`)  

---

## 📅 FINESTRA TEMPORALE ANALIZZATA
{elenco_settimane_md}

---

## 📊 METRICHE CHIAVE PROIETTATE
* **Anomalia SST NINO3.4:** `+{valore_nino:.3f} °C`
* **Tipping Point Dinamico NINO3.4 (Biforcazione):** `+{tipping_point:.2f} °C`
* **Margine / Scostamento dal Tipping Point:** `{distanza_tipping:+.2f} °C`
* **Regime Pliocenico Attivo:** `{regime_attivo}`
* **Saturazione Cella di Hadley / Blocco Africano:** `{rischio_hadley:.1f}%`

---

## 📝 DIAGNOSI SCIENTIFICA E PALEOCLIMATICA (OLLAMA)

{report_text}

---
*Grafico allegato generato automaticamente: `heatwave_diagnostic_rolling.png`*
"""

    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_header)
        
    print(f"      ✔ Report diagnostico Markdown salvato in: {report_md_path}")
    print("\n" + "=" * 75)
    print("   [SUCCESS] PIPELINE ROLLING ESEGUITA CON SUCCESSO!")
    print("=" * 75 + "\n")
    
    # Stampa a schermo dell'anteprima
    print("--- ANTEPRIMA DEL REPORT GENERATO DA OLLAMA ---")
    print(report_text)


if __name__ == "__main__":
    main()
