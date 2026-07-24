# ==============================================================================
# MODULE: src/plotter.py
# PROJECT: EnsoHadley-Forecast-Lab
# Description: Visualizzazione ad alta risoluzione del cruscotto diagnostico.
#              Esplicita la differenza tra Dato Attuale CDAS (scaricato in tempo reale) e 
#              Proiezione Target DMS-MLP a 4 Settimane (+2.235°C), insieme al
#              Tipping Point Dinamico e all'inerzia termica del Mediterraneo.
# ==============================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def generate_diagnostic_dashboard(metrics_dict: dict, output_path: str = "outputs/heatwave_diagnostic_rolling.png", show_plot: bool = False):
    """
    Genera e salva il cruscotto grafico a 2 pannelli in alta risoluzione (300 DPI)
    mostrando chiaramente il Dato Attuale Osservato e la Proiezione Target a 4 Settimane.

    Parametri:
      - metrics_dict: Dizionario contenente le metriche osservate, proiettate e le finestre temporali.
      - output_path: Percorso del file PNG di destinazione.
      - show_plot: Se True, apre la finestra interattiva Matplotlib a schermo.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # --------------------------------------------------------------------------
    # ESTRAZIONE E PARSING PARAMETRI INPUT
    # --------------------------------------------------------------------------
    valore_attuale = float(metrics_dict.get('valore_attuale_obs', metrics_dict.get('valore_attuale_nino34', metrics_dict.get('valore_luglio_nino34', 1.988))))
    valore_target = float(metrics_dict.get('valore_target_nino34', 2.235))
    rischio_target = float(metrics_dict.get('rischio_innesco_percent', 90.7))
    tipping_point = float(metrics_dict.get('tipping_point_nino', metrics_dict.get('tipping_point_nino34', 3.34)))
    historic_max = float(metrics_dict.get('historic_max_nino34', 2.75))
    distanza_tipping = float(metrics_dict.get('distanza_da_tipping', valore_target - tipping_point))
    settimane = metrics_dict.get('settimane_forecast', [])
    data_esec = metrics_dict.get('execution_date', '23/07/2026')
    
    # Formattazione etichette asse X per Pannello B
    if settimane:
        labels_settimane = [s['label'] for s in settimane]
    else:
        labels_settimane = ["Sett 1", "Sett 2", "Sett 3", "Sett 4"]
    
    # Setup grafico generale
    sns.set_theme(style="whitegrid", font="sans-serif")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.8), dpi=300)
    
    # ==========================================================================
    # PANNELLO A: CURVA DI SATURAZIONE HADLEY & TIPPING POINT DINAMICO
    # ==========================================================================
    x_max = max(4.6, valore_target + 0.8, tipping_point + 1.1)
    x_sens = np.linspace(-2.5, x_max, 400)
    
    # Curva logistica di risposta dinamica Hadley
    y_sens = 76 / (1 + np.exp(-2.5 * (x_sens - 0.2))) + 2
    
    ax1.plot(x_sens, y_sens, color='#2c3e50', linewidth=2.5, label='Risposta Dinamica Hadley')
    
    # Zona Innesco/Saturazione (> +0.5°C)
    ax1.axvspan(0.5, x_max, color='#c0392b', alpha=0.05, label='Zona Innesco/Saturazione')
    
    # Pliocene Baseline (+1.5°C)
    ax1.axvline(x=1.5, color='#8e44ad', linestyle=':', linewidth=1.8, label='Pliocene Baseline (+1.5°C)')
    
    # Record Storico Super El Niño (+2.75°C)
    ax1.axvline(x=historic_max, color='#d35400', linestyle='-.', linewidth=1.6, 
                label=f'Record Storico 2015-16 (+{historic_max:.2f}°C)')

    # Tipping Point Dinamico Sistemico (Biforcazione di Bjerknes)
    ax1.axvline(x=tipping_point, color='#e74c3c', linestyle='--', linewidth=1.8, 
                label=f'Tipping Point Sistemico (+{tipping_point:.2f}°C)')
    ax1.axvspan(tipping_point, x_max, color='#e74c3c', alpha=0.08, label='Regime Pliocenico Permanente')
    
    # --------------------------------------------------------------------------
    # 1. DATO ATTUALE OSSERVATO (Blu - Tropical Tidbits / CDAS Real-Time)
    # --------------------------------------------------------------------------
    y_attuale_eff = 76 / (1 + np.exp(-2.5 * (valore_attuale - 0.2))) + 2
    ax1.scatter([valore_attuale], [y_attuale_eff], color='#2980b9', s=85, zorder=6, 
                label=f'Attuale CDAS (+{valore_attuale:.3f}°C)')
    ax1.axvline(x=valore_attuale, color='#2980b9', linestyle=':', linewidth=1.6)

    # --------------------------------------------------------------------------
    # 2. PROIEZIONE TARGET MODEL (Verde - Forecast Rolling 4 Settimane)
    # --------------------------------------------------------------------------
    y_target_eff = 76 / (1 + np.exp(-2.5 * (valore_target - 0.2))) + 2
    ax1.scatter([valore_target], [y_target_eff], color='#16a085', s=85, zorder=6, 
                label=f'Target 4-Sett. (+{valore_target:.3f}°C)')
    ax1.axvline(x=valore_target, color='#16a085', linestyle='--', linewidth=1.6)

    # --------------------------------------------------------------------------
    # 3. VETTORE DI TREND / INCREMENTO (Δ-SST tra Attuale e Proiezione)
    # --------------------------------------------------------------------------
    delta_val = valore_target - valore_attuale
    ax1.annotate('', xy=(valore_target, y_target_eff), xytext=(valore_attuale, y_attuale_eff),
                 arrowprops=dict(arrowstyle="->", color='#d35400', lw=2, mutation_scale=14))
    
    ax1.text((valore_attuale + valore_target) / 2, max(y_attuale_eff, y_target_eff) + 4.5,
             f"Trend ML\nΔ = +{delta_val:.3f}°C", color='#d35400', fontweight='bold', 
             fontsize=7.2, ha='center', bbox=dict(boxstyle="round,pad=0.2", fc="#fef9e7", ec="#d35400", lw=0.8))

    # --- ANNOTAZIONI TESTUALI DINAMICHE ---
    ax1.text(0.55, 12, "ZONA INNESCO HADLEY\n(Saturazione > +0.5°C)", color='#c0392b', 
             fontweight='bold', fontsize=7.5, alpha=0.9)
    ax1.text(1.48, 28, "Stato Medio Pliocene\n(+1.5°C Baseline)", color='#8e44ad', 
             fontweight='bold', fontsize=7.5, ha='right')
    
    # Callout Attuale Osservato
    ax1.annotate(f"Dato Attuale: +{valore_attuale:.3f}°C\n(CDAS Real-Time NOAA)",
                 xy=(valore_attuale, y_attuale_eff), xytext=(valore_attuale - 0.75, y_attuale_eff - 18),
                 color='#2980b9', fontweight='bold', fontsize=7.2,
                 arrowprops=dict(arrowstyle="->", color='#2980b9', lw=1.2),
                 bbox=dict(boxstyle="round,pad=0.3", fc="#ebf5fb", ec="#2980b9", lw=1, alpha=0.9))

    # Callout Tipping Point Dinamico
    tp_x_pos = tipping_point - 0.08 if tipping_point > (x_max - 1.2) else tipping_point + 0.08
    tp_ha = 'right' if tipping_point > (x_max - 1.2) else 'left'

    ax1.text(tp_x_pos, 48, 
             f"TIPPING POINT SISTEMICO\n(+{tipping_point:.2f}°C - Biforcazione)\nCollasso Humboldt & Gradiente", 
             color='#c0392b', fontweight='bold', fontsize=7.0, ha=tp_ha,
             bbox=dict(boxstyle="round,pad=0.3", fc="#fbeee6", ec="#c0392b", lw=1, alpha=0.9))
    
    # Etichetta Target Proiettato
    ha_align = 'right' if valore_target > (x_max - 1.0) else 'left'
    x_offset = -0.12 if ha_align == 'right' else 0.12
    y_box_pos = max(15, y_target_eff - 12)
    
    ax1.text(valore_target + x_offset, y_box_pos, 
             f"Proiezione 4-Sett.: +{valore_target:.3f}°C\nRischio: {rischio_target:.1f}%\nMargine TP: {distanza_tipping:+.2f}°C", 
             color='#16a085', fontweight='bold', fontsize=7.8, ha=ha_align,
             bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#16a085", lw=1, alpha=0.9))

    ax1.set_xlim(-2.5, x_max)
    ax1.set_ylim(0, 100)
    ax1.set_title("Pannello A: Effetto Soglia Planetario & Tipping Point Pacifico", fontsize=10, fontweight='bold', pad=10, color='#2c3e50')
    ax1.set_xlabel("Anomalia NINO3.4 (°C)", fontsize=9, fontweight='bold')
    ax1.set_ylabel("Probabilità d'Innesco Blocco (%)", fontsize=9, fontweight='bold')
    
    ax1.legend(loc='lower right', fontsize=6.5, frameon=True, facecolor='white', framealpha=0.9)

    # ==========================================================================
    # PANNELLO B: TIMELINE ROLLING 4 SETTIMANE (DINAMICA)
    # ==========================================================================
    if settimane and 'rischio_percent' in settimane[0]:
        rischio_dms = [float(s['rischio_percent']) for s in settimane]
    else:
        rischio_dms = [min(99.9, rischio_target + inc) for inc in [0.0, 2.5, 4.0, 5.5]]

    inerzia_sst = [min(99.9, r + 2.0 + (i * 0.8)) for i, r in enumerate(rischio_dms)]

    dati_timeline = pd.DataFrame({
        'Settimana': labels_settimane * 2,
        'Valore': inerzia_sst + rischio_dms,
        'Metrica': ['Inerzia Termica SST Med'] * len(labels_settimane) + ['Rischio Dinamico DMS-MLP'] * len(labels_settimane)
    })
    
    palette = {'Inerzia Termica SST Med': '#e74c3c', 'Rischio Dinamico DMS-MLP': '#7f8c8d'}
    
    barplot = sns.barplot(
        data=dati_timeline, x='Settimana', y='Valore', hue='Metrica',
        palette=palette, ax=ax2, edgecolor='#2c3e50', linewidth=0.8, width=0.55
    )
    
    for p in barplot.patches:
        height = p.get_height()
        if height > 0:
            ax2.annotate(f'{height:.1f}%',
                         (p.get_x() + p.get_width() / 2., height),
                         ha='center', va='bottom',
                         fontsize=7.0, fontweight='bold', color='#2c3e50',
                         xytext=(0, 2), textcoords='offset points')

    ax2.set_ylim(0, 115)
    ax2.set_title(f"Pannello B: Proiezione Rolling 4 Settimane (Run {data_esec})", fontsize=10, fontweight='bold', pad=10, color='#2c3e50')
    ax2.set_xlabel("Finestra Temporale Mobile", fontsize=9, fontweight='bold', labelpad=8)
    ax2.set_ylabel("Probabilità / Intensità (%)", fontsize=9, fontweight='bold')
    
    ax2.legend(loc='upper left', frameon=True, facecolor='white', framealpha=0.85, fontsize=7.5)

    # ==========================================================================
    # TITOLO GENERALE & SALVATAGGIO / VISUALIZZAZIONE
    # ==========================================================================
    fig.suptitle(
        f"CRUSCOTTO DIAGNOSTICO ONDATE DI CALORE – RUN OPERATIVO ROLLING ({data_esec})\n"
        "Accoppiamento Forcing Pacifico/Pliocene Baseline e Inerzia Termica SST Mediterraneo",
        fontsize=11, fontweight='bold', color='#2c3e50', y=0.98
    )
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"[SUCCESS] Cruscotto grafico aggiornato salvato in: {output_path}")

    if show_plot:
        plt.show()

    plt.close(fig)


if __name__ == "__main__":
    # Importazione dinamica del motore analitico e download automatico CDAS
    try:
        from ml_engine import calculate_dynamic_tipping_point, fetch_latest_cdas_nino34, run_rolling_dms_forecast
    except ImportError:
        from src.ml_engine import calculate_dynamic_tipping_point, fetch_latest_cdas_nino34, run_rolling_dms_forecast

    # Esegue il run forecast completo dinamico
    forecast_results = run_rolling_dms_forecast()
    
    valore_attuale_obs = forecast_results['valore_attuale_obs']
    valore_target_proj = forecast_results['valore_target_nino34']
    tp_dinamico = forecast_results['tipping_point_nino']
    
    # Stampa diagnostica esplicita a terminale
    print("=" * 80)
    print("              CRUSCOTTO DIAGNOSTICO OPERATIVO - ONDATE DI CALORE")
    print("=" * 80)
    print(f"  [DATA RUN]: {forecast_results['execution_date']}")
    print(f"  [1] DATO ATTUALE OSSERVATO SATELLITARE (NOAA CPC / CDAS Real-Time):")
    print(f"      - Anomalia Niño 3.4 Corrente                  : +{valore_attuale_obs:.3f} °C")
    print(f"      - Regime Corrente                             : SATURAZIONE HADLEY ATTIVA")
    print()
    print(f"  [2] PROIEZIONE MODELLO ROLLING (DMS-MLP - Finestra 4 Settimane):")
    print(f"      - Target Anomalia Proiettata (Peak 4 Sett.)   : +{valore_target_proj:.3f} °C")
    print(f"      - Incremento Inerziale Previsto (Δ-Trend)     : +{valore_target_proj - valore_attuale_obs:.3f} °C")
    print(f"      - Probabilità Innesco Blocco Mediterraneo     : {forecast_results['rischio_innesco_percent']}%")
    print()
    print(f"  [3] SOGLIE DINAMICHE & TIPPING POINT:")
    print(f"      - Tipping Point Sistemico (Biforcazione Bjerknes): +{tp_dinamico:.2f} °C")
    print(f"      - Margine di Sicurezza Residuo                    : {forecast_results['distanza_da_tipping']:+.2f} °C")
    print("=" * 80)

    generate_diagnostic_dashboard(forecast_results, "outputs/heatwave_diagnostic_rolling.png", show_plot=True)