# 🐄 VetCare AI System: Animal Pregnancy Monitoring and Disease Detection

An AI-powered veterinary decision support system that predicts animal pregnancy status, estimates delivery dates, and detects major reproductive diseases using Machine Learning and Clinical Rule-Based Intelligence.

The system is designed to assist veterinarians and livestock owners by providing accurate, data-driven insights for early diagnosis and better reproductive health management.

---

## 📌 Project Overview

Veterinary healthcare often requires timely diagnosis of pregnancy and infectious reproductive diseases. Manual diagnosis can be time-consuming and may require laboratory testing.

This project integrates Machine Learning models with veterinary clinical knowledge to automate:

- Pregnancy Prediction
- Estimated Delivery Date Prediction
- Brucellosis Detection
- Toxoplasmosis Detection
- Clinical Rule-Based Decision Support

The system improves diagnostic efficiency while supporting veterinarians in making informed clinical decisions.

---

## 🚀 Features

### 🐾 Pregnancy Prediction
Predicts whether an animal is pregnant using multiple physiological and behavioral indicators.

### 📅 Delivery Date Estimation
Calculates the expected delivery date based on mating date and species-specific gestation periods.

### 🦠 Disease Detection
Detects:

- Brucellosis
- Toxoplasmosis
- Normal Healthy Animals

using Machine Learning algorithms.

### 📊 Clinical Rule-Based Diagnosis
Implements expert-defined veterinary rules to provide baseline disease predictions and compare them with Machine Learning models.

### 📈 Performance Evaluation
Evaluates models using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
- Stratified 5-Fold Cross Validation

---

# 🏗 Project Architecture

```
Veterinary Dataset
        │
        ▼
Data Cleaning & Preprocessing
        │
        ▼
Feature Engineering
        │
        ├──────────────┐
        ▼              ▼
Clinical Rules      Machine Learning
                        │
                        ├──────────────┐
                        ▼              ▼
                   XGBoost      XGBoost + Optuna
                        │
                        ▼
                Disease Prediction
                        │
                        ▼
          Performance Evaluation
```

---

# 🛠 Technologies Used

## Programming

- Python

## Machine Learning

- XGBoost
- Optuna
- Autoencoder (Deep Learning)

## Libraries

- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Matplotlib
- Seaborn

## Development Tools

- Jupyter Notebook
- VS Code

---

# 📂 Dataset

The project utilizes veterinary clinical records containing information such as:

- Species
- Age
- Breed
- Body Temperature
- Appetite
- Behaviour Changes
- Pregnancy Indicators
- Fetal Heart Sound
- Brucella Test Result
- Toxoplasma Test Result
- Mating Date
- Estimated Delivery Date

Three datasets were used:

| Dataset | Records |
|----------|--------:|
| Original Dataset | 610 |
| Augmented Dataset | 3,000 |
| Augmented Dataset | 10,000 |

---

# 🤖 Machine Learning Models

## Clinical Rule-Based System

Uses predefined veterinary rules based on expert clinical knowledge.

---

## XGBoost

A gradient boosting model used for multi-class disease classification.

---

## XGBoost + Optuna

Hyperparameter optimization using Optuna to improve model performance.

---

## Autoencoder + XGBoost

An Autoencoder learns compressed feature representations before classification using XGBoost.

---

# 📊 Model Evaluation

Models were evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- Classification Report
- Stratified 5-Fold Cross Validation

---

# 📁 Project Structure

```
VetCare-AI/
│
├── datasets/
│   ├── original_dataset.xlsx
│   ├── augmented_3000.xlsx
│   └── augmented_10000.csv
│
├── notebooks/
│
├── models/
│
├── src/
│   ├── preprocessing.py
│   ├── clinical_rules.py
│   ├── pregnancy_prediction.py
│   ├── disease_detection.py
│   ├── xgboost_model.py
│   ├── optuna_tuning.py
│   ├── autoencoder.py
│   └── evaluation.py
│
├── images/
│
├── requirements.txt
│
└── README.md
```

# 📈 Results

The proposed system demonstrated excellent performance across all datasets.

Evaluation included:

- Original Dataset (610 records)
- Augmented Dataset (3,000 records)
- Augmented Dataset (10,000 records)

using

- XGBoost
- XGBoost + Optuna
- Autoencoder + XGBoost

Performance was measured using Stratified 5-Fold Cross Validation.
<img width="1508" height="853" alt="WhatsApp Image 2026-07-16 at 11 13 07" src="https://github.com/user-attachments/assets/fb4a7295-f544-425f-8fd5-d20d139e735d" />
<img width="1194" height="663" alt="WhatsApp Image 2026-07-16 at 11 13 15" src="https://github.com/user-attachments/assets/80024e2d-c6a8-4664-b604-6c4f1c9eda01" />
<img width="1280" height="705" alt="WhatsApp Image 2026-07-16 at 11 13 34" src="https://github.com/user-attachments/assets/f704d592-e01c-4404-89f8-6a88e878938c" />
<img width="1300" height="788" alt="WhatsApp Image 2026-07-16 at 11 13 43" src="https://github.com/user-attachments/assets/85a6ce93-8db8-4f65-8357-2662e76904f7" />
<img width="1033" height="642" alt="WhatsApp Image 2026-07-16 at 11 14 34" src="https://github.com/user-attachments/assets/65ca2efb-c01c-4221-ad2d-08fbc673a8d2" />

---

# 🔬 Future Scope

- Support additional animal species
- Integration with IoT wearable sensors
- Mobile application deployment
- Cloud-based veterinary monitoring
- Explainable AI (XAI) integration using SHAP
- Real-time farm monitoring dashboards

---

# 👨‍💻 Author

**Sathwik Nomula**

Bachelor of Technology (Computer Science & Data Science)

Machine Learning • Data Science • Artificial Intelligence

---

# ⭐ Acknowledgements

Special thanks to the faculty members, veterinary domain experts, and all contributors who supported this project.

---

## 📜 License

This project is intended for educational and research purposes.
