Activity 1 – Prediction with Supervised Learning Models

Nazim Alperen Akcakaya

Neural and Evolutionary Computing – URV (2025)
📌 Overview

This project implements and evaluates several supervised learning models on the Bike Sharing Dataset (UCI).
The goal is to compare prediction performance and analyse the impact of model design choices such as:

Neural network architecture

Learning rate and momentum

Activation functions

Regularisation (L2, early stopping)

Cross-validation

Ensemble learning

The work includes both custom-built models and scikit-learn models.

📁 Repository Structure
A1-NazimAlperenAkcakaya/
│
├── data/
│   ├── day.csv               # Main Bike Sharing dataset (daily)
│   └── hour.csv              # Not used in this assignment
│
├── NeuralNet.py              # Custom neural network with backpropagation
├── data_and_models.py        # Main experiment runner (MLR, BP, BP-F, ensembles, regularisation)
├── cv_experiments.py         # 5-fold CV for BP model
│
└── README.md                 # Project documentation

🚀 How to Run the Project
Install dependencies:
pip install numpy pandas scikit-learn

Run all experiments (baseline + BP + BP-F + ensembles + regularisation):
python data_and_models.py

Run 5-fold cross-validation for the custom BP network:
python cv_experiments.py

📊 Key Results
1️⃣ MLR-F (baseline)

MSE: 634,351

MAE: 583.02

MAPE: 133.70%

2️⃣ Custom Back-Propagation (BP)

Several network configurations were tested.

Example results:

MSE between 709k–2.7M

MAE between 673–1235

MAPE between 105%–195%

The model is sensitive to hyperparameters and less stable than BP-F.

3️⃣ BP-F (MLPRegressor)

MSE: 567,813

MAE: 548.63

MAPE: 103.88%

Much more stable due to built-in regularisation and better optimisation.

4️⃣ Ensemble Models
Model	MSE	MAE	MAPE
Random Forest	462,389	439.14	144.96%
Gradient Boosting	425,420	449.40	121.22%

Gradient Boosting achieved the best MSE overall.

5️⃣ Regularisation Experiments (BP-F)

Tested L2 (alpha) and Early Stopping.

Best configuration:

alpha = 0.01, early stopping = True

MSE: 383,494

MAE: 446.63

MAPE: 95.21%

Regularisation significantly improved BP-F stability and accuracy.

6️⃣ 5-Fold Cross-Validation (BP)
MSE : 847,175 ± 121,613
MAE : 722.46 ± 34.50
MAPE: 42.36% ± 30.89%


Variation is high → model is sensitive to different partitions.

🧠 Summary of Insights

BP-F (MLPRegressor) outperforms manual BP thanks to built-in optimisation.

Gradient Boosting achieves the best overall prediction accuracy.

Regularisation (L2 + early stopping) is essential for neural network stability.

CV reveals strong variance in the BP model due to dataset complexity.

Custom BP implementation behaves correctly but is not as robust as library models.

📚 Dataset Source

Bike Sharing Dataset
UCI Machine Learning Repository
https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset

✔ Final Notes

This repository follows a natural development workflow with multiple incremental commits:

dataset loading

baseline model

custom BP

sklearn models

ensembles

regularisation

cross-validation

Everything is modular so experiments can be repeated easily.