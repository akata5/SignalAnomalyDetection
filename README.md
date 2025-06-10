# Bayesian Signal Detection and Statistical Analysis

This project performs statistical analysis and signal classification using a combination of classical decision theory and modern machine learning techniques. It includes probability estimation, visualization, threshold-based alarm generation, and supervised learning models trained on real-world-like physiological data.

## 📊 Features

- Probability mass function (PMF) estimation from sampled signals
- Empirical and theoretical thresholding for abnormality detection
- Classical ML classifiers:
  - Maximum Likelihood (ML)
  - Maximum A Posteriori (MAP)
- Modern Machine Learning:
  - **Logistic Regression** (scikit-learn)
  - **Feedforward Neural Network** (PyTorch)
- Evaluation on unseen test patients with:
  - **False Alarm Probability (P_FA)**
  - **Miss Detection Probability (P_MD)**
  - **Overall Error Probability (P_E)**
- Visualization of alarm decisions over time and across models
- Comparison of learned models vs. handcrafted decision rules

## Techniques Used

- Bayesian inference and hypothesis testing
- Classical decision theory (ML/MAP)
- Logistic regression with scikit-learn
- Neural networks with PyTorch
- Empirical vs. theoretical statistical modeling
- Data visualization (Matplotlib)
- Scientific computing (NumPy, SciPy)
- Data preprocessing and normalization (StandardScaler)

## Project Structure

- `SignalDetection_CleanedFinal.ipynb` – Main notebook containing:
  - Data sampling and PMF estimation
  - Alarm generation logic
  - ML and MAP classifiers
  - Logistic Regression & Neural Network training
  - Model evaluation and comparison

## 🚀 Getting Started

1. Clone this repo and navigate to the directory
2. Install dependencies:

```bash
pip install numpy scipy matplotlib scikit-learn torch
