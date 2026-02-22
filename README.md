# Proactive Flight Rerouting via Deep Neural Networks and SHAP Explainability

**Author:** William Zhou — Plano West Senior High School  
**Mentor:** Independent Research (Science Fair, Dallas Regional / ISEF Track)  
**Status:** Active — Publication under revision at *National High School Journal of Science*

---

## Overview

Current Air Traffic Control (ATC) systems are **reactive**: controllers respond to delays only after they occur. This project builds a **proactive** pipeline that predicts flight delays before departure and automatically recommends optimal alternative routes — with full explainability for FAA-facing transparency requirements.

The system consists of three cascaded stages:

```
[17 Input Features] → Stage 1: Delay Predictor DNN → Stage 2: Route Optimizer DNN → Stage 3: SHAP Explainability
                                  (128→64→32→2)              (256→128→64→4)              (KernelExplainer)
```

### Key Results

| Metric | Value |
|--------|-------|
| Delay Prediction Accuracy | **61.32%** (balanced) |
| Inference Speed | **< 100ms** |
| Speed vs. Manual ATC | **3,000×** faster |
| Features Processed | **17** simultaneously |
| Top Delay Predictor (SHAP) | Airport Congestion (0.1035) |

---

## Motivation

Flight delays cost the U.S. economy **$33 billion annually** and affect **71 million passengers** per year (FAA, 2023). Current systems rely on expert intuition under time pressure. This project shifts the paradigm from reactive to proactive through:

- **Physics-grounded modeling** via the [OpenAP](https://openap.dev/) aircraft performance library
- **Explainable AI** via SHAP (Shapley Additive Explanations) to eliminate black-box decisions
- **Realistic synthetic data** modeled on US-CONUS high-density ARTCC sectors

---

## Repository Structure

```
flight_rerouting/
│
├── src/
│   ├── dnn_flight_rerouting.py        # Core DNN architecture (from scratch, NumPy only)
│   └── train_realistic_models.py      # Full training pipeline with OpenAP integration
│
├── data/
│   ├── realistic_flight_dataset.csv   # 10,000 flights with OpenAP performance features
│   └── realistic_rerouting_dataset.csv # Alternative route comparisons (3 options/flight)
│
├── models/
│   ├── realistic_delay_model.pkl      # Trained delay prediction model
│   └── realistic_reroute_model.pkl    # Trained route optimization model
│
├── results/
│   ├── delay_feature_importance.csv   # SHAP feature rankings — delay model
│   ├── reroute_feature_importance.csv # SHAP feature rankings — rerouting model
│   └── shap_plots/
│       ├── delay_shap_bar.png         # Feature importance bar chart (delay)
│       ├── delay_shap_summary.png     # Beeswarm SHAP summary (delay)
│       ├── reroute_shap_bar.png       # Feature importance bar chart (rerouting)
│       └── reroute_shap_summary.png   # Beeswarm SHAP summary (rerouting)
│
├── requirements.txt
└── README.md
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the core DNN (synthetic data, no OpenAP required)
```bash
python src/dnn_flight_rerouting.py
```

### 3. Run the full realistic training pipeline
```bash
# Requires OpenAP: pip install openap
python src/train_realistic_models.py
```

---

## Technical Architecture

### Stage 1 — Delay Predictor
- **Input:** 17 features (geographic, temporal, operational, environmental)
- **Architecture:** 128 → 64 → 32 → 2 (Softmax)
- **Training:** Mini-batch SGD, lr=0.001, batch=64, epochs=50
- **Regularization:** He initialization, gradient clipping (±1.0), Z-score normalization
- **Output:** Delay probability [0, 1]

### Stage 2 — Route Optimizer
- **Input:** 19 features (includes delay probability from Stage 1 + alternative route metrics)
- **Architecture:** 256 → 128 → 64 → 4 (Softmax)
- **Output:** Optimal route selection (Original / Weather-Optimized / Wind-Optimized / Congestion-Optimized)

### Stage 3 — SHAP Explainability
- **Method:** KernelSHAP (model-agnostic, game-theory-based)
- **Background:** 100-sample training subset
- **Validation:** Top SHAP features align with FAA domain expertise (e.g., Airport Congestion ranked #1)

### Physics Integration (OpenAP)
Aircraft performance data is computed using the [OpenAP](https://openap.dev/) library (Sun et al., 2020), providing:
- Fuel consumption models per aircraft type (A320, B737, B738, A321, B788, A359)
- Cruise altitude and speed envelopes
- Drag/thrust physics for realistic route cost calculations

---

## SHAP Results

Top features by mean absolute SHAP value for the **delay prediction model**:

| Rank | Feature | SHAP Value | Aviation Validation |
|------|---------|-----------|---------------------|
| 1 | Airport Congestion | 0.1035 | Primary delay driver per FAA data |
| 2 | Previous Delay | 0.0554 | Cascading/propagation effect |
| 3 | Aircraft Age | 0.0460 | Maintenance correlation |
| 4 | Origin Weather | 0.0405 | Terminal area weather impact |

✅ **No spurious correlations detected** — model learned real aviation patterns.

---

## Dataset

Synthetic dataset of **10,000 flights** generated with:
- Realistic US-CONUS route pairs (90 high-traffic sector pairs)
- OpenAP aircraft performance models for 6 aircraft types
- Historical weather pattern distributions
- FAA ARTCC congestion modeling

> **Note:** Data is synthetic but physics-grounded. It is not derived from real airline operational records.

---

## Citations

- Sun, J., Hoekstra, J. M., & Ellerbroek, J. (2020). *OpenAP: An Open-Source Aircraft Performance Model for Air Transportation Studies and Simulations*. Aerospace, 7(8), 104.
- Hoekstra, J. M., & Ellerbroek, J. (2016). *BlueSky ATC Research Simulator*. ICNS Conference.
- Lundberg, S. M., & Lee, S. I. (2017). *A unified approach to interpreting model predictions*. NeurIPS.

---

## License

MIT License — free to use with attribution.
