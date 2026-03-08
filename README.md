# 🛰️ GNSS Anti-Spoofing Detection
### NyneOS Technologies × Kaizen 2026 × EES × ARIES IIT Delhi

> *Detecting GPS spoofing attacks using a Transformer-based deep learning model with multi-satellite early fusion — achieving Val F1 scores up to **0.9798** across 5-fold cross-validation.*

---

## 📡 The Problem

Global Navigation Satellite Systems (GNSS) are the invisible backbone of modern civilization — powering aviation, maritime navigation, autonomous drones, financial transaction timestamps, and national infrastructure. Yet their signals are inherently fragile: low-power, unencrypted, and broadcast openly across the sky.

**Spoofing** exploits this vulnerability. A malicious transmitter sends fabricated satellite signals that trick a GNSS receiver into computing a false position, velocity, or time (PVT) estimate — silently, with no visible error to the operator.

```
Real World:   Satellite ──────────────────────────► Receiver ✅
Spoofed:      Satellite   Spoofer ──fake signals──► Receiver ❌
                          (single ground source)
```

The consequences range from aircraft being guided off-course, to drones being hijacked mid-flight, to financial systems receiving corrupted timestamps. Detection is non-trivial — a well-crafted spoofing attack can seamlessly replace legitimate signals before the receiver notices any jump.

This project builds an AI-driven detector that identifies spoofing by learning temporal anomalies and multi-satellite signal inconsistencies — the subtle fingerprints that betray a fake signal source.

---

## 🔬 Research Foundation

Two papers directly shaped our architecture and feature engineering:

### Paper 1 — Zelinka et al. (2025)
**"Deep Sequence-to-Sequence Models for GNSS Spoofing Detection"** — *arXiv:2510.19890*
*University of West Bohemia, Czech Republic*

This paper benchmarks four model configurations (LSTM vs. Transformer × early fusion vs. late fusion) across targeted and regional spoofing scenarios. Key findings that influenced our design:

| Finding | Our Design Response |
|---|---|
| MHA (Transformer) outperforms LSTM across all scenarios | Used Transformer encoder as the core architecture |
| Early fusion outperforms late fusion | Concatenated all 8 satellite channels per timestep into one vector before feeding the model |
| Raw pseudorange suffers from gradient explosion due to magnitude | Applied StandardScaler normalization across all features |
| Signal presence indicators prevent missing-channel noise | Added `Signal_Valid_Mask` feature (PRN > 0) |
| Best total error rate: **0.16%** with MHA + Early Fusion | Validated our architecture choice |

> *"Early fusion enables a direct combination of signals from different satellites during processing"* — Zelinka et al.

### Paper 2 — Li et al. (2025)
**"C/N0 Analysis-Based GPS Spoofing Detection with Variable Antenna Orientations"** — *arXiv:2510.16229*
*Cornell University & Embry-Riddle Aeronautical University*

This paper establishes a critical physical intuition: **spoofed signals lack spatial diversity.** Because all fake signals originate from a single transmitter, they behave as if every satellite is in the same location. Under real conditions, C/N0 values vary naturally across satellites based on geometry; under spoofing, they become unnaturally uniform.

| Observation | Our Feature Response |
|---|---|
| Spoofed signals show flat C/N0 across all PRNs (single point source) | `CNO_Spatial_Std`: std of CN0 across channels per timestamp |
| Real signals vary temporally with atmosphere/geometry | `CNO_Temporal_Std`: rolling std of CN0 per satellite over time |
| Per-PRN variation: ~4–15 dB real sky vs ~0.5–3 dB spoofed | These features directly capture this measurable contrast |

---

## ⚙️ Feature Engineering

Every feature was chosen based on physical reasoning about what spoofing does to GNSS signals.

### Raw Features (from problem statement)

| Feature | Physical Meaning | Why it Detects Spoofing |
|---|---|---|
| `Carrier_Doppler_hz` | Frequency shift due to relative motion | Spoofed signals often have incorrect or static Doppler profiles |
| `Carrier_phase` | Accumulated phase of carrier wave | Spoofing causes sudden phase discontinuities |
| `EC`, `LC`, `PC` | Early / Late / Prompt correlator outputs | Shape distortion in the correlation triangle reveals signal manipulation |
| `PIP`, `PQP` | Signal quality metrics | Anomalous quality scores flag suspect signal conditions |
| `TCD` | Code delay | Spoofing alters timing delay characteristics |
| `TOW` | Time of week from satellite | Timing mismatches between satellite and receiver clock indicate spoofing |
| `CN0` | Carrier-to-noise ratio (signal strength) | Spoofed signals show abnormal strength patterns |
| `Pseudorange_m` | Apparent distance to satellite | Spoofing creates geometric inconsistencies in the pseudorange solution |

### Engineered Features (paper-inspired)

```python
# Early-Minus-Late discriminator: quantifies correlator asymmetry
df['EML_Shape'] = df['EC'] - df['LC']

# Signal validity mask: prevents empty channels from corrupting fusion
df['Signal_Valid_Mask'] = (df['PRN'] > 0).astype(float)

# Cross-satellite CNO variance: low variance = likely single-source (spoofed)
# Inspired by Li et al. — spoofed signals lack spatial diversity
df['CNO_Spatial_Std'] = df.groupby('time')['CN0'].transform('std')

# Per-satellite temporal CNO stability: spoofed signals are unnaturally stable
# Rolling std over 5 timesteps per PRN
df['CNO_Temporal_Std'] = df.groupby('PRN')['CN0'].transform(
    lambda x: x.rolling(window=5, min_periods=1).std()
)
```

**Total feature vector per channel:** 14 features
**Total input per timestep:** 8 channels × 14 features = **112-dimensional fused vector**

---

## 🏗️ Model Architecture

### `GNSS_Spoof_Detector` — Transformer Encoder with Early Fusion

```
┌─────────────────────────────────────────────────────┐
│  Input: (batch, seq_len=64, 112)                    │
│         8 satellite channels × 14 features          │
│                        ↓                            │
│  Linear Projection  →  d_model = 128                │
│                        ↓                            │
│  Sinusoidal Positional Encoding                     │
│                        ↓                            │
│  ┌─────────────────────────────┐                    │
│  │  TransformerEncoderLayer ×2 │                    │
│  │  · 8 attention heads        │                    │
│  │  · dropout = 0.1            │                    │
│  │  · residual connections     │                    │
│  └─────────────────────────────┘                    │
│                        ↓                            │
│  Linear(128→64) → ReLU → Linear(64→1) → Sigmoid    │
│                        ↓                            │
│  Output: (batch, seq_len, 1)                        │
│  Per-timestep spoofing confidence score [0, 1]      │
└─────────────────────────────────────────────────────┘
```

### Hyperparameters

| Parameter | Value | Rationale |
|---|---|---|
| `SEQ_LEN` | 64 | Captures enough temporal context for Doppler drift and phase walk to manifest |
| `D_MODEL` | 128 | Matches the embedding dimension in Zelinka et al.'s best-performing model |
| `N_HEADS` | 8 | Allows the model to attend to different feature subspaces simultaneously |
| `NUM_LAYERS` | 2 | Lightweight — prevents overfitting on the available ~1,741 training sequences |
| `BATCH_SIZE` | 32 | Standard for sequence models of this size |
| `LEARNING_RATE` | 1e-3 | AdamW; implicit weight decay aids generalization |

### Why Transformer over LSTM?

Zelinka et al. tested both on identical data. MHA-based models achieved **0.16% total error** vs LSTM's **0.21%**. The attention mechanism can simultaneously attend to any timestep in the window — rather than relying on a hidden state that decays over distance. This matters here because spoofing artifacts don't always appear at the most recent timestep: a Doppler anomaly at t-20 should still influence the prediction at t-0.

### Why Early Fusion?

In late fusion, each satellite is processed independently and results are aggregated at the end. But the key signature of spoofing — that *all satellites appear to come from the same source* — is a **cross-satellite relationship**. Early fusion concatenates all 8 channels before the first attention layer, allowing the model to directly learn "are these channels behaving too similarly to be real?"

---

## 🏋️ Training Methodology

### 5-Fold Shuffled Cross-Validation

With ~1,741 labeled sequences, a single train/val split risks high variance. We used 5-fold shuffled K-Fold CV (`random_state=42`) to ensure every sequence contributes to both training and validation, and class balance is maintained across folds.

```
Full Dataset (1,741 sequences)
├── Fold 1: Train 1392 | Val 349
├── Fold 2: Train 1393 | Val 348
├── Fold 3: Train 1393 | Val 348
├── Fold 4: Train 1393 | Val 348
└── Fold 5: Train 1393 | Val 348
```

### Early Stopping (Patience = 3)

Each fold trains for up to 20 epochs but stops if Val F1 does not improve for 3 consecutive epochs. The best model checkpoint per fold is saved and reloaded for test inference — ensuring we never submit a degraded model.

```python
optimizer = AdamW(lr=1e-3)
criterion = BCELoss()      # binary cross-entropy per timestep
metric    = weighted_F1   # directly matches the competition evaluation metric
```

### Ensemble Strategy

After all 5 folds complete, **soft voting** is used: the raw sigmoid confidence scores from each fold's best model are averaged, then thresholded at 0.5 for the final binary label.

```
Final Confidence = (fold1_score + fold2_score + ... + fold5_score) / 5
Final Label      = 1 if Final Confidence > 0.5 else 0
```

Soft voting is preferable to hard voting here because it preserves calibration — a near-threshold prediction that all 5 folds agree on at 0.51 is treated differently from one where folds are split 3/2 and the average is 0.48.

---

## 📊 Training Results

| Fold | Best Val F1 | Stopped At Epoch | Notes |
|------|------------|-----------------|-------|
| 1 | **0.9776** | 10 | Steady improvement through epoch 7, plateau then early stop |
| 2 | 0.9651 | 11 | Minor instability at epoch 10 (Val F1 dipped to 0.83), recovered to 0.9567 before stopping |
| 3 | **0.9798** | 5 | Fastest convergence — best single-fold result |
| 4 | 0.9617 | 9 | Most conservative fold, consistent improvement |
| 5 | 0.9727 | 4 | Converged in just 4 epochs |
| **Avg** | **0.9714** | — | Consistent high performance across all folds |

All 5 folds converged well within the 20-epoch budget. Early stopping triggered in every single fold, confirming the features are highly discriminative and the model saturates quickly rather than overfitting.

The Fold 2 dip at epoch 10 (Train F1: 0.93, Val F1: 0.83) followed by partial recovery is consistent with a transient gradient instability — the early stopping mechanism correctly identified it and terminated 1 epoch later, protecting the best saved checkpoint.

---

## 📁 Repository Structure

```
.
├── spoofed.ipynb          # Full pipeline: preprocessing → model → training → inference
├── submission_final.csv   # Final ensembled predictions on the test set
└── README.md              # This document
```

---

## 🚀 Reproducing Results

1. Place `train.csv` and `test.csv` at the input paths in the notebook (or update file paths).
2. Run all cells in `spoofed.ipynb` sequentially.
3. The notebook will:
   - Load and clean raw CSV, handling non-numeric garbage rows via `pd.to_numeric(errors='coerce')`
   - Engineer all 14 features per channel including paper-inspired CNO variance features
   - Normalize using `StandardScaler` (fit on train data only, applied to test)
   - Reshape into 8-channel × 14-feature early-fusion format
   - Run 5-fold cross-validation with early stopping
   - Ensemble soft predictions across all 5 folds
   - Save `submission_final.csv`

**Dependencies:** `pandas`, `numpy`, `torch`, `scikit-learn`

---

## 📤 Output Format

| Column | Type | Description |
|---|---|---|
| `time` | float | Receiver timestamp from the test dataset |
| `Spoofed` | int (0 or 1) | Binary prediction — `1` = spoofed, `0` = genuine |
| `Confidence` | float [0, 1] | Ensemble-averaged sigmoid score representing P(spoofed) |

The `Confidence` column always represents the model's probability that the sample **is spoofed**, regardless of the final binary label. A `Confidence` of 0.32 with `Spoofed=0` correctly reads as: *"32% confident this is a spoofed signal — classified as genuine."*

---

## 🔭 Future Directions

- **Second-difference pseudorange features**: Zelinka et al. showed that Δ²(pseudorange) correlates with receiver acceleration and amplifies spoofing-induced anomalies while removing the slow-varying baseline magnitude. This is a strong candidate for a next iteration.
- **Learnable quantized embeddings**: Rather than a simple linear projection into `d_model`, quantizing input values into a probability distribution over learnable codebook entries (as in Zelinka et al.) could improve gradient flow for high-magnitude features like `Pseudorange_m`.
- **Dual-antenna C/N0 asymmetry**: Li et al. propose that two spatially separated antennas produce distinct C/N0 responses to real satellites but nearly identical responses to a spoofed single-source signal — a hardware-level signature that could complement this model as a secondary detection channel.
- **Targeted vs. regional spoofing heads**: Zelinka et al. found that targeted and regional attacks have very different detection profiles (regional is harder mid-attack but easy at boundaries). A multi-task head that distinguishes attack type alongside detection could improve robustness.
