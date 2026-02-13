# 🔐 UNSW-NB15 Intrusion Detection System

Binary Classification: **Normal vs Attack**  
Dataset: **UNSW-NB15**  
Assignment: ML Assignment 2  

---

## 📌 Overview

This project implements multiple machine learning models to detect network intrusions using the UNSW-NB15 cybersecurity dataset.

The goal is simple:

- `0 → Normal`
- `1 → Attack`

The pipeline covers the full end-to-end ML workflow:

- Data preprocessing  
- Model training  
- Model evaluation & comparison  
- Model serialization  
- Streamlit deployment  

This is not just model training — it includes real deployment and evaluation.

---

## 📊 Dataset Information

**Total Samples:** 257,673  
**Total Features:** 35 (after preprocessing)

### Class Distribution

| Label | Count |
|-------|-------|
| Attack (1) | 164,673 |
| Normal (0) | 93,000 |

The dataset is moderately imbalanced, so evaluation was not based on accuracy alone.

---

## 🧠 Models Implemented

The following classifiers were trained:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest  
- XGBoost  

All models were trained using preprocessing pipelines that include:

- StandardScaler (for numerical features)  
- OneHotEncoder (for categorical features)  

---

## 📈 Evaluation Metrics

Models were compared using:

- **Accuracy**
- **AUC (ROC)**
- **Precision**
- **Recall**
- **F1 Score**
- **MCC (Matthews Correlation Coefficient)**

### Why MCC?

Intrusion detection is a security problem with class imbalance.  
**MCC** was used as the primary metric because it evaluates TP, TN, FP, and FN together in a balanced way.

Accuracy alone is not reliable in imbalanced datasets.

---

## 🏆 Best Performing Model

Based on the final run:

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|-------|----------|------|----------|--------|------|------|
| XGBoost | 0.9383 | 0.9891 | 0.9540 | 0.9493 | 0.9516 | 0.8666 |
| Decision Tree | 0.9326 | 0.9737 | 0.9590 | 0.9344 | 0.9465 | 0.8558 |
| Random Forest | 0.9315 | 0.9872 | 0.9293 | 0.9663 | 0.9474 | 0.8505 |
| KNN | 0.8965 | 0.9631 | 0.9149 | 0.9240 | 0.9194 | 0.7749 |
| Logistic Regression | 0.8709 | 0.9486 | 0.8570 | 0.9579 | 0.9046 | 0.7175 |
| Naive Bayes | 0.5052 | 0.6669 | 0.9996 | 0.2258 | 0.3684 | 0.3083 |

> **XGBoost** achieved the strongest overall performance (highest MCC and AUC).

Example confusion matrix from training:

|               | Predicted Normal | Predicted Attack |
|---------------|-----------------|-----------------|
| Actual Normal | 17092           | 1508            |
| Actual Attack | 1670            | 31265           |

The model achieves strong recall while maintaining high precision — important for intrusion detection systems.

## 🔍 Model Performance Observations

| Model | Observation about Model Performance |
|-------|-------------------------------------|
| Logistic Regression | Achieved high recall (0.9579) but lower precision (0.8570), indicating more false positives. Performs reasonably well but underfits compared to ensemble methods. Suitable as a strong linear baseline. |
| Decision Tree | High precision (0.9590) and strong overall balance. Slightly lower recall than Random Forest. Prone to overfitting but performs well on structured tabular data. |
| KNN | Moderate performance across all metrics. Sensitive to feature scaling and high dimensionality. Computationally expensive at inference time. |
| Naive Bayes | Extremely high precision (0.9996) but very poor recall (0.2258). Fails to detect many attack cases. Strong class bias assumption limits performance on complex data. |
| Random Forest | Excellent recall (0.9663) and strong F1 (0.9474). Good bias–variance tradeoff. Robust and stable ensemble performer. |
| XGBoost | Best overall model. Highest MCC (0.8666) and AUC (0.9891). Strong balance between precision and recall. Most suitable for deployment in intrusion detection setting. |

---

## 💾 Saved Model Files

All trained models are serialized as `.pkl` files inside the `models/` directory.

| Model | Size |
|-------|------|
| xgboost.pkl | 0.30 MB |
| random_forest.pkl | 3.84 MB |
| decision_tree.pkl | 0.43 MB |
| knn.pkl | 20.03 MB |
| logistic_regression.pkl | 0.01 MB |
| naive_bayes.pkl | 0.01 MB |

File sizes were verified to meet deployment constraints.

---

## 🌐 Streamlit Web Application

The project includes a fully interactive Streamlit app.
🚀 [Live Streamlit App](https://unsw-nb15-intrusion-detection.streamlit.app/)


### Features

- Model selection dropdown  
- Upload custom test CSV  
- Automatic metric computation  
- Confusion matrix visualization  
- ROC curve visualization  
- Downloadable sample test dataset  

Users can upload a CSV file containing a `label` column to evaluate performance instantly.

---

## 📂 Repository Structure

UNSW-NB15-Intrusion-Detection-ML/<br>
&nbsp;&nbsp;│<br>
&nbsp;&nbsp;├── app.py<br>
&nbsp;&nbsp;├── requirements.txt<br>
&nbsp;&nbsp;├── README.md<br>
&nbsp;&nbsp;├── model_results.csv<br>
&nbsp;&nbsp;├── test_sample.csv<br>
&nbsp;&nbsp;│<br>
&nbsp;&nbsp;└── models/<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── logistic_regression.pkl<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── decision_tree.pkl<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── knn.pkl<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── naive_bayes.pkl<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;├── random_forest.pkl<br>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;└── xgboost.pkl<br>

---

## ⚙️ Requirements

Main dependencies:

- streamlit  
- pandas  
- numpy  
- scikit-learn==1.6.1  
- xgboost==3.1.3  
- matplotlib  
- seaborn  
- joblib  

Install locally with:

```bash
pip install -r requirements.txt
