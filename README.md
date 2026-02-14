# 🔐 UNSW-NB15 Intrusion Detection System
 
Dataset: [**UNSW-NB15**](https://www.kaggle.com/datasets/dhoogla/unswnb15/)<br>
Task: **Binary Classification (Normal vs Attack)** 

Name: **Ayan Ahmad** <br>
BITS ID: **2025AA05215** <br>
Assignment: ML Assignment 2  <br>

---

## 1️⃣ Overview

In this project, I used the UNSW-NB15 cybersecurity dataset to build a network intrusion detection system using multiple machine learning models.

The task is a binary classification problem:

- **0 → Normal**
- **1 → Attack**

The workflow followed was end-to-end:

- Data preprocessing  
- Training multiple classification models  
- Evaluating and comparing their performance  
- Saving the trained models  
- Deploying the final interface using Streamlit  

For this assignment, six different classification models were trained on the same dataset, compared using metrics such as Accuracy, AUC, F1-score, and MCC, and deployed through a Streamlit app so they can be tested on new data.

---

## 2️⃣ Dataset Information

**Dataset Name:** UNSW-NB15  
**Dataset Link:** https://www.kaggle.com/datasets/dhoogla/unswnb15  
**Total Samples:** 257,673  
**Total Features:** 35 (after removing the multi-class column)

### Class Distribution

| Label | Count |
|-------|-------|
| Attack (1) | 164,673 |
| Normal (0) | 93,000 |

The dataset contains more attack samples than normal samples, so accuracy alone is not enough to judge performance. Along with accuracy, AUC, F1-score, and MCC were also used to better understand how well the models handle both classes.

---

## 3️⃣ Models Implemented

The following classification models were trained on the same dataset for fair comparison:

- Logistic Regression  
- Decision Tree  
- K-Nearest Neighbors (KNN)  
- Naive Bayes  
- Random Forest (Ensemble)  
- XGBoost (Ensemble)  

Each model was trained using the same train–test split and evaluated using the same metrics.

### Preprocessing Pipeline

Before training the models, a preprocessing pipeline was applied:

- **StandardScaler** for numerical features  
- **OneHotEncoder** for categorical features  

This preprocessing step was included inside a `Pipeline` along with each classifier so that identical transformations are applied during both training and testing.

---

## 4️⃣ Evaluation Metrics

Each model was evaluated using:

- Accuracy  
- AUC (ROC Score)  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Why MCC?

Since this is an intrusion detection problem, the dataset is not perfectly balanced. There are more attack samples than normal ones, so accuracy alone can be misleading.

MCC takes into account all four outcomes of the confusion matrix (TP, TN, FP, FN) and provides a more balanced performance measure, especially when classes are uneven.

For this reason, accuracy alone was not used to compare models.

---

## 5️⃣ Model Performance Comparison 📊

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------|----------|------|-----------|--------|------|------|
| XGBoost | 0.9383 | 0.9891 | 0.9540 | 0.9493 | 0.9516 | 0.8666 |
| Decision Tree | 0.9326 | 0.9737 | 0.9590 | 0.9344 | 0.9465 | 0.8558 |
| Random Forest | 0.9315 | 0.9872 | 0.9293 | 0.9663 | 0.9474 | 0.8505 |
| KNN | 0.8895 | 0.9576 | 0.9074 | 0.9210 | 0.9142 | 0.7591 |
| Logistic Regression | 0.8713 | 0.9487 | 0.8579 | 0.9572 | 0.9048 | 0.7182 |
| Naive Bayes | 0.5052 | 0.6669 | 0.9996 | 0.2258 | 0.3684 | 0.3083 |

From the results, the ensemble models performed the best overall. Both Random Forest and XGBoost achieved strong performance across most metrics.

Among all models, **XGBoost achieved the highest MCC and AUC while maintaining a good balance between precision and recall.** Decision Tree also performed well, but the ensemble versions were more consistent.

Naive Bayes struggled on this dataset, particularly in recall and MCC, indicating that it was not well-suited for this feature space.

Based on these results, **XGBoost was selected as the final model for deployment.**

---

## 6️⃣ Confusion Matrix (XGBoost) 🔎

| | Predicted Normal | Predicted Attack |
|---|-----------------|-----------------|
| **Actual Normal** | 17,092 | 1,508 |
| **Actual Attack** | 1,670 | 31,265 |

From the confusion matrix, the model correctly identifies a large number of attack samples while keeping false negatives relatively low.

Since missing attacks (false negatives) can be critical in intrusion detection, this balance between precision and recall makes XGBoost suitable for deployment.

---

## 7️⃣ Model Performance Observations 📈

| Model | Observation |
|--------|-------------|
| Logistic Regression | High recall (0.9579) but lower precision (0.8570), leading to more false positives. Serves as a solid linear baseline but underperforms compared to ensemble methods. |
| Decision Tree | High precision (0.9590) with balanced performance. Performs well on structured tabular data but can overfit. |
| KNN | Moderate performance. Sensitive to high dimensionality after one-hot encoding and computationally expensive. |
| Naive Bayes | Extremely high precision (0.9996) but very poor recall (0.2258). Misses many attack cases due to strong independence assumptions. |
| Random Forest | Excellent recall (0.9663) and strong F1 score. Provides stable ensemble performance. |
| XGBoost | Best overall performer with highest MCC (0.8666) and AUC (0.9891). Strong balance between precision and recall. |

---

## 8️⃣ Saved Model Files 💾

All trained models were saved as `.pkl` files inside the `model/` directory.

| File | Size |
|-------|-------|
| xgboost.pkl | 0.30 MB |
| random_forest.pkl | 3.84 MB |
| decision_tree.pkl | 0.43 MB |
| knn.pkl | 21.41 MB |
| logistic_regression.pkl | 0.01 MB |
| naive_bayes.pkl | 0.01 MB |

The KNN file is larger because it stores training samples internally, whereas models like Logistic Regression and Naive Bayes store only learned parameters.

---

## 9️⃣ Streamlit Web Application 🌐

A Streamlit application was built and deployed for interactive testing.

**Streamlit Link:**  
https://unsw-nb15-intrusion-detection.streamlit.app/

### Features

- Model selection dropdown  
- Upload custom test CSV file  
- Automatic evaluation metric computation  
- Confusion matrix visualization  
- ROC curve visualization  
- Downloadable sample test dataset  

Users can upload a CSV file containing a `label` column to evaluate performance instantly.

---

## 🔟 Repository Structure 📂

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

## 1️⃣1️⃣ Requirements ⚙️

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
