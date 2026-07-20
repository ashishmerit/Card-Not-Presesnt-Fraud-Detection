# Real-Time Card-Not-Present Fraud Detection

## Executive Summary
This project builds a robust classification engine to detect fraudulent credit card transactions in a highly imbalanced dataset. Unlike standard classification problems, fraud detection requires optimizing for the **Precision-Recall Trade-off**. 

This capstone explores data separability, applies rigorous scaling, and utilizes Stratified Cross-Validation to evaluate various Support Vector Machine (SVM) configurations. The ultimate goal is to maximize the Precision-Recall Area Under the Curve (PR-AUC) to minimize the operational costs of false positives (declining legitimate customers) and false negatives (missing actual fraud).

## Business Problem
In e-commerce and FinTech, fraudulent transactions are extremely rare compared to legitimate ones. However, each fraud event triggers chargebacks and loss of merchant trust. 
* **False Positives:** Wrongfully declining a legitimate customer causes friction, cart abandonment, and potential churn.
* **False Negatives:** Missing a fraudulent transaction results in direct financial loss.
The model must balance these two competing business risks.

## Dataset
* **Source:** [Kaggle: Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* **Description:** Transactions made by European cardholders in September 2013.
* **Features:** 28 anonymized numerical features (V1-V28) resulting from a PCA transformation, plus `Time` and `Amount`.
* **Target:** `Class` (0 = Legitimate, 1 = Fraud). The dataset is highly imbalanced, with frauds accounting for only 0.172% of all transactions.

## Methodology
1. **Exploratory Data Analysis (EDA):** Quantified the severe class imbalance and visualized feature distributions (using logarithmic scales for monetary amounts) to ensure mathematical separability.
2. **Preprocessing:** Applied `RobustScaler` to handle extreme outliers in transaction amounts and created a 10% stratified subsample to allow for computationally feasible SVM training while preserving the exact fraud ratio.
3. **Feature Selection:** Utilized an `ExtraTreesClassifier` to compute Mean Decrease in Impurity (MDI). *Finding:* Reducing features actually degraded linear model performance and increased training time, proving that subtle behavioral indicators in high-dimensional space are necessary for this dataset.
4. **Model Training & Comparison:** Evaluated Linear, Radial Basis Function (RBF), and Polynomial kernels.
5. **Hyperparameter Tuning:** Conducted a `GridSearchCV` on the RBF kernel to optimize the regularization parameter ($C$) and the kernel coefficient ($\gamma$), balancing the bias-variance trade-off.

## Key Findings & Performance Metrics
| Model Configuration | Training Time (s) | PR-AUC |
| :--- | :--- | :--- |
| **Baseline (Linear)** | 8.51 | 0.6732 |
| **Reduced Features (Linear)** | 19.15 | 0.8508 |
| **Default RBF** | 10.26 | 0.3481 |
| **Polynomial (Degree 3)** | 9.28 | 0.9080 |
| **Tuned RBF (C=10, gamma=0.01)** | 99.01 | 0.7588 |

*Note: Training times reflect the 10% stratified subsample used during the tuning phase.*

## Final Production Recommendation
**Deploy the Polynomial Kernel SVM.**

The Baseline Linear model achieved excellent recall but suffered from a severe precision deficit (11%), meaning it would flag 9 legitimate transactions for every 1 true fraud caught—an unacceptable operational cost in customer support. While hyperparameter tuning rescued the RBF kernel's precision (jumping to 50%), it forced recall down to 70%, allowing too much financial loss.

The Polynomial kernel offers the most mathematically sound compromise, achieving the highest overall PR-AUC (0.6880) without the computational overhead of the grid-searched RBF. It successfully captures the non-linear boundaries of consumer behavior.

## Tech Stack
* **Language:** Python 3
* **Libraries:** Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / VS Code
