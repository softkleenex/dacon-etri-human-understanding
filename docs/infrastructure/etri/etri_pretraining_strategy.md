# ETRI Lifelog 2018-2020 Pre-training Strategy

## 1. Schema Analysis: Dacon 2024 vs ETRI 2020

### Target Mapping (Survey Data)
The Dacon 2024 targets are binary values (`0` or `1`) mapped from the original 5-point Likert scales collected via daily surveys. The ETRI 2020 dataset contains the **raw 1-5 scale values**.

- **ETRI 2020 (`user_survey_2020.csv`)**:
  - `sleep` (1~5): Sleep quality
  - `amCondition` (1~5): Morning condition
  - `pmEmotion` (1~5): Afternoon emotion
  - `pmStress` (1~5): Afternoon stress
  - `pmFatigue` (1~5): Afternoon fatigue
- **Dacon 2024 (`ch2026_metrics_train.csv`)**:
  - `Q1`: (Derived from sleep quality)
  - `Q2`: (Derived from amCondition)
  - `Q3`: (Derived from pmEmotion/Stress/Fatigue)
  - `S1~S4`: Objective sleep metrics (Derived from sleep duration, HR, etc.)

**Pre-training Advantage**: We can train our CNN-Transformer to directly predict the raw 1-5 scales from the 2018-2020 data as a **Proxy Task**, then fine-tune it on the Dacon binary targets. This creates an incredibly powerful Domain-Specific Foundation Model.

### Sensor Data Mapping
The `user_sleep_2020.csv` contains pre-extracted features like `wakeupduration`, `lightsleepduration`, `deepsleepduration`, `hr_average`, `hr_min`, `hr_max`.
The massive `.7z` files contain the raw minute-by-minute sensor logs (WiFi, BLE, GPS, Pedo, HR).

## 2. Kaggle-compatible Pre-training Pipeline

To utilize this 50GB+ dataset, we must process it externally or upload it as a separate Kaggle Dataset.

### Phase 1: Local Preprocessing
1. **Extract**: Use `7z x user11-12.7z` to extract the raw CSVs locally.
2. **Standardize**: Run a script that converts the 2018-2020 raw schemas into the exact same parquet format used by Dacon 2024 (e.g., `ch2025_mLight.parquet`).
3. **Upload to Kaggle**: Create a new Kaggle Dataset named `softkleenex/etri-lifelog-2018-2020-pretrain`.

### Phase 2: Foundation Model Training
1. In `dl_trainer.py`, add a `--pretrain` mode.
2. Train the CNN-Transformer on the 50GB dataset predicting the raw survey scores (MSE Loss).
3. Save the weights as `etri_foundation_model.ckpt`.

### Phase 3: Dacon Fine-Tuning
1. Load `etri_foundation_model.ckpt` into our Dacon Single Holdout pipeline.
2. Freeze the CNN feature extractor, replace the final classification head to output 7 binary targets (BCEWithLogitsLoss).
3. Fine-tune for 10-20 epochs on the Dacon 450-day data.
