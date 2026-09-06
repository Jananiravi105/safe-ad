# SAFE-AD — Spacecraft Anomaly Detection System

## Overview
Real-time multivariate time series anomaly detection 
on NASA SMAP/MSL spacecraft telemetry data.

Extended base paper: Pattern Recognition, Elsevier 2024 (IF: 7.5)

## Results
- F1 Score = 0.9524 (Channel M-6)
- ROC-AUC  = 0.9873
- Recall   = 1.0 (zero missed anomalies)
- Inference = 2.97 seconds

## 4 Novel Contributions
1. Multivariate — 25 sensors simultaneously
2. 3-Module Ensemble — STFT + IsoForest + Transformer
3. Sensor Attribution — top 5 responsible sensors
4. Adaptive Thresholding — no manual tuning

## Technologies
Python, PyTorch, Scikit-learn, Streamlit, 
SciPy, NumPy, Pandas

## Live Demo
[Click here to open SAFE-AD Dashboard](https://share.streamlit.io/user/jananiravi105)
