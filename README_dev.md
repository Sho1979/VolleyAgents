# VolleyAgents - Documentazione Sviluppo

## Test di Regressione

### Test di Regressione Principale

**⚠️ IMPORTANTE: Questo test NON deve mai rompersi quando modifichi agenti, HeadCoach o MasterCoach.**

Il test di regressione verifica che il sistema continui a produrre risultati corretti sulla finestra temporale 16:50-26:54 del video Millennium.

#### Comando di Regressione

**Versione repo-relative (consigliata per chi clona il progetto):**

```bash
python -m tools.video_pipeline.eval_rallies_gt \
  --gt tools/video_pipeline/ground_truth/gt_millennium_16m50_26m54_verified.json \
  --pred tools/video_pipeline/ground_truth/rallies_1010_1600.json \
  --iou 0.5
```

> **Nota**: Copia il file predetto (`rallies_1010_1600.json`) nella cartella `tools/video_pipeline/ground_truth/` per "freezarlo" nel repository.

**Esempio (setup locale Cristian):**

```bash
python -m tools.video_pipeline.eval_rallies_gt `
  --gt tools/video_pipeline/ground_truth/gt_millennium_16m50_26m54_verified.json `
  --pred "C:/Users/Administrator/OneDrive - CRISTIAN PERANI/2025/Volley/Test1/rallies_1010_1600.json" `
  --iou 0.5
```

> **Nota sui comandi**: 
> - `\` è per shell tipo bash (WSL, macOS, Linux)
> - In PowerShell usa il backtick `` ` `` a fine riga oppure tutto su una riga

#### Risultati Attesi

- **Ground truth rally**: 19
- **Rally predetti**: 19
- **Match (TP)**: 19
- **False Negative**: 0
- **False Positive**: 0
- **Precision**: 1.000 (100%)
- **Recall**: 1.000 (100%)
- **Offset medio start**: 0.00 s
- **Offset medio end**: 0.00 s

#### Quando Eseguire il Test

Esegui questo test ogni volta che:
- Modifichi gli agenti (AudioAgent, MotionAgent, ServeAgent, etc.)
- Modifichi HeadCoach o MasterCoach
- Cambi parametri di configurazione
- Modifichi la logica di fusione o validazione

#### Cosa Fare se il Test Fallisce

1. **Verifica che il file predetto esista** (path dipende dalla versione usata)
2. **Rigenera i rally** se necessario usando la GUI o lo script di segmentazione
3. **Analizza i False Positive/Negative** per capire cosa è cambiato
4. **Ripristina le modifiche** se il test è critico e non puoi permetterti regressioni

---

## Ground Truth

### File GT Verificati

- **`gt_millennium_16m50_26m54_verified.json`**: 19 rally verificati manualmente per la finestra 16:50-26:54
  - Formato: `{"id": int, "start": float, "end": float, "side": str}`
  - Usato per il test di regressione principale

- **`gt_millennium_16m50_26m54.json`**: GT originale con 23 rally (formato MM:SS)
  - Formato: `{"serve_time_str": "MM:SS", "point_time_str": "MM:SS", ...}`
  - Mantenuto per compatibilità

### Creare Nuova GT

Usa lo script `make_gt_from_rallies.py`:

```bash
python -m tools.video_pipeline.make_gt_from_rallies \
  --input results/rallies_XXXX.json \
  --output tools/video_pipeline/ground_truth/gt_new.json \
  --keep-ids keep_ids.txt
```

### Aggiungere una Nuova Finestra di Partita

Checklist completa per aggiungere una nuova finestra GT:

1. **In `va_desktop.py` imposta `t_start` / `t_end`** per la nuova finestra
2. **Analizza e Esporta rally** in `results/rallies_XXXX.json`
3. **Verifica le clip MP4 generate** (segna eventuali ID sbagliati)
4. **Genera la GT**:
   - **Se tutti i rally sono buoni:**
     ```bash
     python -m tools.video_pipeline.make_gt_from_rallies \
       --input results/rallies_XXXX.json \
       --output tools/video_pipeline/ground_truth/gt_XXXX_verified.json
     ```
   - **Se alcuni rally sono da escludere:**
     - Crea `tools/video_pipeline/keep_ids_XXXX.txt` con gli ID 1-based (uno per riga)
     - Lancia lo script con `--keep-ids keep_ids_XXXX.txt`
5. **Aggiungi il file GT al repository** e documenta nel README

---

## Struttura del Progetto

```
VolleyAgents/
├── volley_agents/          # Package principale
│   ├── agents/             # Agenti specializzati
│   ├── core/               # Core (Timeline, Rally, Event)
│   ├── fusion/             # HeadCoach, MasterCoach, Voting
│   └── io/                 # I/O (video, config)
├── tools/                  # Tool di utilità
│   └── video_pipeline/     # Pipeline video
│       ├── eval_rallies_gt.py      # Valutazione GT
│       ├── make_gt_from_rallies.py  # Generazione GT
│       └── ground_truth/    # File GT verificati
├── scripts/                # Script eseguibili
│   ├── run_demo.py         # Demo CLI
│   └── va_desktop.py       # GUI Desktop
└── tests/                  # Test unitari
```

---

## Workflow di Sviluppo

1. **Modifica codice** (agenti, coach, etc.)
2. **Esegui test di regressione** (comando sopra)
3. **Verifica che Precision/Recall siano ≥ 0.95** (idealmente 1.00)
4. **Se il test fallisce**: analizza e correggi
5. **Commit** solo se il test passa

---

## Note Tecniche

### ScoreboardAgentV3 (OCR LED)

- **Strategia**: preprocessing adattivo per LED (canale rosso/verde, equalizzazione, morfologia), EasyOCR come backend principale, template matching come fallback e detector di cambio frame come ulteriore segnale.
- **Pipeline**: ROI → preprocessing → EasyOCR → fallback template matching → stabilizzazione temporale → evento `SCORE_CHANGE`.
- **Calibrazione ROI**: prima di eseguire l'OCR è obbligatorio salvare una ROI reale del tabellone (x, y, w, h). Puoi usare la GUI (`va_desktop.py` → bottone "ROI tabellone") oppure il nuovo tool CLI:

  ```bash
  python -m tools.scoreboard.calibrate_scoreboard_v3 \
      --video "Millennium Bienno.mp4" \
      --time 1010.0 \
      --output "scoreboard_config.json"
  ```

  Il JSON generato può essere ricaricato così:

  ```python
  from volley_agents.agents.scoreboard_v3 import ScoreboardAgentV3, ScoreboardConfigV3
  from volley_agents.io.config import load_scoreboard_roi_config

  roi = load_scoreboard_roi_config("scoreboard_config.json", expected_video="Millennium Bienno.mp4")
  if roi is None:
      raise RuntimeError("Scoreboard ROI mancante")

  x, y, w, h = roi
  scoreboard_agent = ScoreboardAgentV3(
      ScoreboardConfigV3(x=x, y=y, w=w, h=h, led_color="red"),
      enable_logging=True,
  )
  ```

- **Configurazione GUI**: in `va_desktop.py` seleziona la ROI del tabellone, abilita la checkbox della lettura tabellone e (opzionale) salva i frame di debug nella cartella di output.
- **Dipendenze**: installa EasyOCR (trascina con sé PyTorch) prima di lanciare la GUI:

  ```bash
  pip install easyocr --break-system-packages
  ```

### HeadCoach - Regole Attive

**⚠️ Questa configurazione di HeadCoach è quella validata dal test di regressione principale (19/19 rally corretti). Se modifichi queste regole, esegui sempre il test prima di fare commit.**

- **Serve obbligatoria**: Ogni rally deve contenere almeno un `SERVE_START`
- **Durata minima**: 0.3s (permette ACE/errori servizio)
- **Durata massima**: 45.0s
- **Gap minimo**: 1.0s tra rally consecutivi
- **Validazione attività palla**: Rally > 0.5s devono avere azioni palla (HIT, ATTACK, etc.)

### Formato JSON Rally

**Input (esportato dalla GUI)**:
```json
{
  "start": 1020.2,
  "end": 1028.0,
  "side": "right"
}
```

**Output GT**:
```json
{
  "id": 1,
  "start": 1020.2,
  "end": 1028.0,
  "side": "right"
}
```

---

## Troubleshooting

### Test di Regressione Fallisce

1. Verifica che il file predetto sia aggiornato
2. Controlla i log per errori durante la segmentazione
3. Verifica che gli agenti stiano producendo eventi corretti
4. Controlla i parametri di configurazione

### Precision/Recall Bassa

- **Precision bassa**: Troppi False Positive → rafforza filtri in HeadCoach
- **Recall bassa**: Troppi False Negative → rilassa filtri o migliora agenti

### Offset Elevati

- **Offset start**: Modifica `_refine_segment_boundaries` in HeadCoach
- **Offset end**: Verifica logica di fine rally (WHISTLE_END, MOTION_GAP)

