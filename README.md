# Activity 1 – Prediction with Supervised Learning Models  
**Nazim Alperen Akcakaya**  
Master’s Programme – Neural and Evolutionary Computation  
Universitat Rovira i Virgili (URV), 2025  

---

## 📘 Overview

This project evaluates several supervised learning models on the **Bike Sharing Dataset (hour.csv)** using regression techniques.  
The task is to predict the **hourly number of rented bicycles** based on temporal, environmental, and seasonal features.

The assignment includes:

- A **custom neural network** implemented from scratch (Back-Propagation + Momentum)  
- A scikit-learn neural network (**MLPRegressor / BP-F**)  
- A baseline **Multiple Linear Regression (MLR-F)** model  
- **Ensemble models** (Random Forest, Gradient Boosting)  
- **L2 regularisation and early stopping** experiments  
- **5-fold Cross-Validation** with the custom BP network  

All models share the same preprocessing pipeline (standardisation + one-hot encoding).

---

## 📁 Repository Structure

A1-NazimAlperenAkcakaya/
│
├── data/
│ └── hour.csv # Dataset used for all experiments (17,379 samples)
│
├── NeuralNet.py # Custom feed-forward neural network implementation
├── data_and_models.py # Main experiment runner
├── cv_experiments.py # 5-fold CV with custom BP model
├── plot_results.py # Learning curves + BP scatter plot
├── plot_other_models.py # BP-F + Random Forest scatter plots
│
└── README.md

---

## ⚙️ How to Run the Experiments

### Install dependencies:

```bash
pip install numpy pandas scikit-learn matplotlib

Run all supervised models:
python data_and_models.py

Run 5-fold Cross-Validation for custom BP:
python cv_experiments.py

Generate plots (learning curves + scatter plots):
python plot_results.py
python plot_other_models.py

Results Summary
MLR-F Baseline
MSE: 18,727  
MAE: 103.14  
MAPE: 353.49%
As expected, a linear model cannot capture the nonlinear structure of the dataset.

Custom Back-Propagation (BP)
Multiple architectures were tested. The best configuration:
Architecture: [38, 32, 16, 1]  
MSE: 7,342.91  
MAE: 43.70  
MAPE: 51.90%

Observations:
Works correctly
But suffers from vanishing gradients and online SGD instability
Underestimates high-demand hours (visible in scatter plot)

BP-F (MLPRegressor)
makefile
Copy code
MSE: 2,289.85  
MAE: 31.29  
MAPE: 45.64%

Benefits:
Adaptive Adam optimiser
Batch learning
Built-in regularisation
Much stronger than manual BP.

Ensemble Models (Optional Part 3)
Model	                MSE	        MAE	    MAPE
Random Forest	        1,722.57	24.84	32.92%
Gradient Boosting	    3,611.82	40.72	79.40%
➡ Random Forest is the best-performing model in this study.

Regularisation Experiments (BP-F)
Tested α ∈ {0.0001, 0.001, 0.01, 0.1} with/without early stopping.
Best configuration:
alpha = 0.001
early_stopping = False
MSE: 2,109.09
MAE: 29.25
MAPE: 42.21%
Regularisation improves stability and reduces overfitting.

5-Fold Cross-Validation (custom BP)
MSE: 7,778.04 ± 434.41  
MAE: 45.61 ± 1.56  
MAPE: 47.06% ± 4.22%
Small standard deviations → stable generalisation, but performance still limited by the manual BP approach.

Key Insights
Random Forest is the most accurate model for this dataset.
BP-F (MLPRegressor) significantly outperforms the custom BP network.
The custom BP implementation is correct but suffers from:
 vanishing gradients
 online SGD noise
 difficulty modelling nonlinear peaks
Regularisation (L2 + early stopping) is essential for improving neural network performance.
Feature scaling + one-hot encoding greatly improve convergence for all neural models.

Dataset Reference
Bike Sharing Dataset
UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

Final Notes
This repository follows a clean and modular workflow:
Data preprocessing
Baseline linear regression
Custom BP implementation
BP-F model
Ensemble models
Regularisation
Cross-validation
Plot generation
Every component is isolated for reproducibility and clarity.

Author:
Nazim Alperen Akcakaya
Neural and Evolutionary Computation – URV (2025)
