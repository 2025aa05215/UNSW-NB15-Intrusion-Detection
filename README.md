# 🔐 UNSW-NB15 Intrusion Detection System
 
Dataset: [**UNSW-NB15**](https://www.kaggle.com/datasets/dhoogla/unswnb15/)<br>
Task: **Binary Classification (Normal vs Attack)** 

Name: **Ayan Ahmad** <br>
BITS ID: **2025AA05215** <br>
Assignment: ML Assignment 2  <br>

---

## 1️⃣ Problem Statement

The objective of this project is to build a machine learning based intrusion detection system using the UNSW-NB15 dataset.

This is a binary classification task:

- 0 → Normal Network Traffic  
- 1 → Attack Traffic  

The goal is to compare multiple classification models and determine which performs best for detecting malicious network activity.

---

## 2️⃣ Dataset Description

**Dataset Name:** UNSW-NB15  
**Source:** Kaggle  
**Total Samples:** 257,673  
**Total Features:** 35 (after removing the multi-class column)

### Class Distribution

| Label | Count |
|-------|-------|
| Attack (1) | 164,673 |
| Normal (0) | 93,000 |

The dataset contains more attack samples than normal samples.  
Because of this imbalance, accuracy alone is not sufficient for evaluation.

---

## 3️⃣ Models Used

The following six models were implemented on the same dataset:

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbor (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble)  

All models were trained using:

- StandardScaler for numerical features  
- OneHotEncoder for categorical features  
- Stratified train-test split  

---

## 4️⃣ Model Performance Comparison 📊

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---------------|----------|------|-----------|--------|------|------|
| Logistic Regression | 0.8713 | 0.9487 | 0.8579 | 0.9572 | 0.9048 | 0.7182 |
| Decision Tree | 0.9326 | 0.9737 | 0.9590 | 0.9344 | 0.9465 | 0.8558 |
| KNN | 0.8895 | 0.9576 | 0.9074 | 0.9210 | 0.9142 | 0.7591 |
| Naive Bayes | 0.5052 | 0.6669 | 0.9996 | 0.2258 | 0.3684 | 0.3083 |
| Random Forest (Ensemble) | 0.9315 | 0.9872 | 0.9293 | 0.9663 | 0.9474 | 0.8505 |
| XGBoost (Ensemble) | 0.9383 | 0.9891 | 0.9540 | 0.9493 | 0.9516 | 0.8666 |

---

## 5️⃣ Model Performance Observations 📈

| ML Model Name | Observation about model performance |
|---------------|-------------------------------------|
| Logistic Regression | High recall but lower precision. Works as a good linear baseline but underperforms compared to ensemble methods. |
| Decision Tree | Strong precision and balanced performance. Performs well on structured data but can overfit. |
| KNN | Moderate performance. Sensitive to high dimensionality and computationally heavier than other models. |
| Naive Bayes | Very high precision but very poor recall. Misses many attack cases due to independence assumptions. |
| Random Forest (Ensemble) | Strong recall and F1 score. Provides stable ensemble performance. |
| XGBoost (Ensemble) | Best overall performer with highest MCC and AUC. Maintains a strong balance between precision and recall. |

---

## 6️⃣ Confusion Matrix (XGBoost) 🔎

| | Predicted Normal | Predicted Attack |
|---|-----------------|-----------------|
| **Actual Normal** | 17,092 | 1,508 |
| **Actual Attack** | 1,670 | 31,265 |

XGBoost was selected for deployment based on its overall balanced performance across evaluation metrics.

---

## 7️⃣ Streamlit Deployment 🌐

Live App:  
https://unsw-nb15-intrusion-detection.streamlit.app/

The Streamlit app includes:

- CSV dataset upload option  
- Model selection dropdown  
- Automatic evaluation metric display  
- Confusion matrix visualization  
- ROC curve visualization  
- Downloadable sample test dataset  

---

## 8️⃣ Repository Structure 📂

```
UNSW-NB15-Intrusion-Detection/
├── app.py
├── requirements.txt
├── README.md
├── model_training.ipynb
├── model_results.csv
├── test_sample.csv
└── model/
    ├── logistic_regression.pkl
    ├── decision_tree.pkl
    ├── knn.pkl
    ├── naive_bayes.pkl
    ├── random_forest.pkl
    └── xgboost.pkl
```

---

## 9️⃣ Requirements ⚙️

Main dependencies:

- streamlit  
- pandas  
- numpy  
- scikit-learn (1.6.1)  
- xgboost (3.1.3)  
- matplotlib  
- seaborn  
- joblib  

To install locally:

```
pip install -r requirements.txt
```

---
