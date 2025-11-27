# Activity 1 – Prediction with Supervised Learning Models  
**Nazim Alperen Akcakaya**  
Master’s Programme – Neural And Evolutionary Computation 
Universitat Rovira i Virgili (URV), 2025

---

## 📘 Overview

This project evaluates several supervised learning models on the **Bike Sharing Dataset** (UCI).  
The objective is to predict the **daily count of rented bicycles** using environmental, seasonal, and temporal features, and to study how different learning strategies affect predictive performance.

The assignment includes:

- A **custom feed-forward neural network** implemented from scratch (Back-Propagation + Momentum)  
- A scikit-learn neural model (**MLPRegressor / BP-F**)  
- A baseline **Multiple Linear Regression (MLR-F)** model  
- **Ensemble methods** (Random Forest and Gradient Boosting)  
- **Regularisation experiments** (L2 penalty, early stopping)  
- **5-fold Cross-Validation** using the custom BP network  

All experiments use the same preprocessing pipeline to ensure comparable results.

---

## 📁 Repository Structure

A1-NazimAlperenAkcakaya/
│
├── data/
│ ├── day.csv # Dataset used for the assignment (731 samples)
│ └── hour.csv # Larger dataset (not used in this report)
│
├── NeuralNet.py # Custom neural network implementation (from scratch)
├── data_and_models.py # Main experiment runner
├── cv_experiments.py # 5-fold CV for the custom BP model
│
└── README.md # Documentation


---

## ⚙️ How to Run the Experiments

### Install dependencies:

```bash
pip install numpy pandas scikit-learn

Run all models (MLR-F, BP, BP-F, Ensembles, Regularisation):
python data_and_models.py

Run the 5-fold cross-validation experiment:
python cv_experiments.py

📊 Results Summary
1️⃣ MLR-F Baseline
MSE: 634,351
MAE: 583.02
MAPE: 133.70%

2️⃣ Custom Back-Propagation (BP)
Several architectures were tested:
MSE: 709k – 2.7M
MAE: 673 – 1235
MAPE: 105% – 196%
The model is highly sensitive to hyperparameters and exhibits instability due to online updates.

3️⃣ BP-F (MLPRegressor)
MSE: 567,813
MAE: 548.63
MAPE: 103.88%

This model clearly improves upon the manual BP version thanks to better optimisation and built-in regularisation.

4️⃣ Ensemble Models (Optional Part 3)
Model	             MSE	    MAE	     MAPE
Random Forest	     462,389	439.14	 144.96%
Gradient Boosting	 425,420	449.40	 121.22%
➡ Gradient Boosting achieves the best predictive accuracy overall.

5️⃣ Regularisation Experiments (Optional Part 1)
Best configuration:
alpha = 0.01
early_stopping = True
Results:
MSE: 383,494
MAE: 446.63
MAPE: 95.21%
Regularisation notably improves generalisation.

6️⃣ 5-Fold Cross-Validation (Optional Part 2)
Using the custom BP model:
MSE: 847,175 ± 121,613
MAE: 722.46 ± 34.50
MAPE: 42.36% ± 30.89%
Fold variability reveals the instability of the online BP learning rule.

🧠 Key Insights
BP-F (MLPRegressor) reliably outperforms the manual BP implementation.
Gradient Boosting provides the lowest MSE among all models.
Regularisation (L2 + early stopping) is essential for stable neural network learning.
The custom BP algorithm works correctly, but its performance is highly dependent on learning rate, momentum, and architecture.
Ensemble models effectively capture nonlinear seasonal effects in the dataset.

📚 Dataset Reference
Bike Sharing Dataset (UCI Machine Learning Repository):
https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

✔️ Final Notes
This repository was developed following a clear, incremental workflow:
Dataset preprocessing
Baseline regression
Custom BP implementation
BP-F model
Ensemble learners
Regularisation studies
Cross-validation
All modules are independent, making the experimentation process clean and reproducible.

Author: Nazim Alperen Akcakaya
Neural And Evolutionary Computation – URV (2025)

