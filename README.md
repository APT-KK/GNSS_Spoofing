# GNSS Anti-Spoofing Detection — Hackathon Submission
**NyneOS Technologies × Kaizen 2026 × EES × ARIES IIT Delhi**

---

## Problem Understanding

Global Navigation Satellite Systems (GNSS) are critical infrastructure for aviation, maritime navigation, autonomous drones, logistics, and financial systems. Because GNSS signals are inherently low-power and broadcast in open environments, they are vulnerable to **spoofing attacks** — where a malicious transmitter sends fabricated signals to trick a receiver into computing incorrect position, velocity, or timing (PVT) estimates.

This project addresses the challenge of building an AI-driven detector capable of identifying spoofing attempts by analyzing signal behavior and temporal patterns across GNSS channels.

The task is a **binary sequence classification problem** on an imbalanced dataset, evaluated using **Weighted F1 Score**.

---

## Research References

Two papers informed the design decisions of this solution:

1. **Zelinka et al. (2025) — *Deep Sequence-to-Sequence Models for GNSS Spoofing Detection*** (`arXiv:2510.19890`): This paper demonstrates that Transformer-based architectures with **early fusion** of multi-satellite inputs outperform LSTM and late-fusion strategies, achieving an error rate as low as 0.16%. It also introduces the use of the **second difference of pseudorange** to capture subtle temporal anomalies, and proposes using **signal presence indicator masks** to handle missing satellite channels.

2. **Li et al. (2025) — *C/N0 Analysis-Based GPS Spoofing Detection with Variable Antenna Orientations*** (`arXiv:2510.16229`): This paper establishes that spoofed signals lack spatial diversity — all channels appear to originate from a single point source, resulting in abnormally low variance in C/N0 across satellites. This motivated our engineered features for **cross-channel CNO spatial variance** and **per-satellite CNO temporal stability**.

---

## Feature Engineering

Raw features were selected based on the physical meaning of each signal attribute as described in the problem statement:

| Feature Group | Columns Used | Rationale |
|---|---|---|
| Carrier Dynamics | `Carrier_Doppler_hz`, `Carrier_phase` | Spoofed signals show abnormal Doppler shifts and sudden phase discontinuities |
| Correlator Shape | `EC`, `LC`, `PC` | Shape distortion in early/late correlators is a classic spoofing indicator |
| Signal Quality | `PIP`, `PQP` | Anomalous quality metrics flag suspect signals |
| Timing | `TCD`, `TOW` | Spoofing alters code delay and timing characteristics |
| Standalone | `CN0`, `Pseudorange_m` | CNO patterns and geometric inconsistencies in pseudorange |

**Engineered features (paper-inspired):**

- **`EML_Shape`** (`EC - LC`): The Early-Minus-Late discriminator quantifies correlator asymmetry, a well-known indicator of signal shape distortion caused by spoofing.

- **`Signal_Valid_Mask`**: A binary indicator (`PRN > 0`) marking whether a channel carries a real satellite signal. Inspired by Zelinka et al.'s use of presence indicators to handle missing signals without corrupting the model's attention.

- **`CNO_Spatial_Std`**: Standard deviation of CN0 values across all active channels at each timestamp. Motivated by Li et al.'s finding that spoofed signals — originating from a single transmitter — exhibit abnormally low spatial variance in signal strength across satellites.

- **`CNO_Temporal_Std`**: Rolling standard deviation (window=5) of CN0 per satellite over time. Spoofed signals are often unnaturally stable temporally; this feature captures that anomaly.

All features were normalized using **StandardScaler** (fit on training data, applied to test data).

---

## Model Architecture

### Early Fusion Transformer (`GNSS_Spoof_Detector`)

Inspired by the MHA + Early Fusion configuration in Zelinka et al. (which achieved the best results in their experiments), we implemented a **Transformer Encoder** that processes multi-channel GNSS data as a sequence.

**Fusion strategy:** At each timestep, all 8 satellite channels are concatenated into a single flat vector — an "early fusion" approach that allows the model to reason about inter-satellite relationships from the very first layer.

```
Input: (batch, seq_len=64, 8_channels × 14_features = 112)
  → Linear projection → d_model = 128
  → Sinusoidal Positional Encoding
  → TransformerEncoder (2 layers, 8 heads, dropout=0.1)
  → Linear(128 → 64) → ReLU → Linear(64 → 1) → Sigmoid
Output: (batch, seq_len, 1)  — per-timestep spoofing confidence
```

**Key design choices:**
- `d_model = 128`, `N_HEADS = 8`, `NUM_LAYERS = 2` — kept lightweight to avoid overfitting on the available training data
- **Sequence length of 64 timesteps** — captures enough temporal context for pattern anomalies to manifest without being computationally prohibitive
- **BCELoss** — appropriate for the binary per-timestep classification objective
- **Sigmoid output** — produces a calibrated confidence score; threshold of 0.5 used for final binary prediction

---

## Training Methodology

### 5-Fold Shuffled Cross-Validation with Early Stopping

To maximize generalization on the limited dataset (~1,741 total sequences), we used **5-fold shuffled K-Fold cross-validation** with `random_state=42`.

**Training configuration:**
- Optimizer: `AdamW`, `lr=1e-3`
- Max epochs: 20
- Early stopping patience: 3 epochs (no improvement in Val F1)
- Best model per fold saved and reloaded for test inference

**Ensemble strategy:** Test predictions from all 5 folds are averaged (soft voting on confidence scores), then thresholded at 0.5 for the final binary label. This reduces variance and produces more robust predictions than any single fold.

### Training Results Summary

| Fold | Best Val F1 | Epochs Run |
|------|------------|------------|
| 1 | 0.9776 | 10 (early stop) |
| 2 | 0.9651 | 11 (early stop) |
| 3 | 0.9798 | 5 (early stop) |
| 4 | 0.9617 | 9 (early stop) |
| 5 | 0.9727 | 4 (early stop) |

All folds converged quickly, with the best models saved well within the first 10 epochs. Early stopping successfully prevented overfitting in all cases.

---

## Justification of Design Decisions

| Decision | Justification |
|---|---|
| Transformer over LSTM | Zelinka et al. showed MHA-based models consistently outperform LSTM across both targeted and regional spoofing scenarios |
| Early fusion over late fusion | Early fusion allows direct cross-satellite reasoning from the first layer; Zelinka et al. found it superior to late fusion |
| CNO spatial variance feature | Li et al. demonstrated that spoofed signals lack spatial diversity — all satellite signals appear to come from one source, resulting in compressed CNO variance |
| Shuffled K-Fold CV | Given class imbalance, shuffled splits ensure each fold sees a representative mix of spoofed and genuine samples |
| 5-fold ensemble | Averaging soft confidence scores across folds reduces prediction variance and handles the imbalanced class distribution more gracefully than a single model |
| Weighted F1 metric | Directly aligned with the competition's evaluation metric; handles class imbalance without rewarding majority-class bias |
| Sequence windowing (seq_len=64) | Temporal patterns like Doppler drift, phase walk, and CNO stabilization require a sufficient history window to detect; 64 timesteps balances context depth with computational efficiency |

---

## Repository Structure

```
.
├── spoofed.ipynb          # Main notebook: preprocessing, model, training, inference
├── submission_final.csv   # Final predictions on the test set
└── README.md              # This file
```

---

## How to Reproduce

1. Place `train.csv` and `test.csv` in the expected input path (or update the file paths in the notebook).
2. Run all cells in `spoofed.ipynb` sequentially.
3. The notebook will:
   - Clean and engineer features
   - Run 5-fold cross-validation with early stopping
   - Ensemble predictions across folds
   - Save `submission_final.csv`

**Dependencies:** `pandas`, `numpy`, `torch`, `scikit-learn`

---

## Output Format

The submission file `submission_final.csv` contains:

| Column | Description |
|---|---|
| `time` | Receiver timestamp (from test dataset) |
| `Spoofed` | Binary prediction: `1` = spoofed, `0` = genuine |
| `Confidence` | Ensemble-averaged sigmoid confidence score (0–1) |
