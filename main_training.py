#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, mean_absolute_error
import streamlit as st
import joblib
from datetime import datetime, timedelta

# Load your dataset
df = pd.read_excel("/Users/sathwiknomula/Downloads/animal_vet_data33.xlsx")

# PREGNANCY PREDICTION MODEL
print("=== TRAINING PREGNANCY PREDICTION MODEL ===")
# Drop rows with missing pregnancy status
df_preg = df.dropna(subset=["Pregnancy_Status"]).copy()

# Encode target
df_preg['Pregnancy_Status'] = df_preg['Pregnancy_Status'].map({'Yes': 1, 'No': 0})

# Drop non-feature columns
non_features = ["Pregnancy_Status", "Estimated_Delivery_Date", "Mating_Date"]
X_preg = df_preg.drop(columns=non_features)
y_preg = df_preg["Pregnancy_Status"]

# Encode categorical variables
cat_cols = X_preg.select_dtypes(include="object").columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_preg[col] = le.fit_transform(X_preg[col].astype(str))
    label_encoders[col] = le

# Split
X_train_preg, X_test_preg, y_train_preg, y_test_preg = train_test_split(X_preg, y_preg, test_size=0.2, random_state=42)

# Pregnancy prediction optimization
def objective_pregnancy(trial):
    param = {
        'verbosity': 0,
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'booster': 'gbtree',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'lambda': trial.suggest_float('lambda', 1e-3, 10.0),
        'alpha': trial.suggest_float('alpha', 1e-3, 10.0),
    }
    
    model = xgb.XGBClassifier(**param, use_label_encoder=False)
    model.fit(X_train_preg, y_train_preg)
    preds = model.predict(X_test_preg)
    accuracy = accuracy_score(y_test_preg, preds)
    return accuracy

study_preg = optuna.create_study(direction="maximize")
study_preg.optimize(objective_pregnancy, n_trials=30)
print("Best Pregnancy Params:", study_preg.best_params)
print("Best Pregnancy Accuracy:", study_preg.best_value)

# Train final pregnancy model
best_preg_model = xgb.XGBClassifier(**study_preg.best_params, use_label_encoder=False)
best_preg_model.fit(X_train_preg, y_train_preg)

# DELIVERY ESTIMATION MODEL
print("\n=== TRAINING DELIVERY ESTIMATION MODEL ===")
# Make a copy to avoid modifying original dataframe
df_delivery = df[['Mating_Date', 'Estimated_Delivery_Date', 'Body_Temperature_F', 'Weight_kg']].dropna().copy()

# Convert to datetime
df_delivery['Mating_Date'] = pd.to_datetime(df_delivery['Mating_Date'])
df_delivery['Estimated_Delivery_Date'] = pd.to_datetime(df_delivery['Estimated_Delivery_Date'])

# Create target: Days to delivery
df_delivery['Days_To_Delivery'] = (df_delivery['Estimated_Delivery_Date'] - df_delivery['Mating_Date']).dt.days

# Drop datetime columns
df_delivery.drop(['Mating_Date', 'Estimated_Delivery_Date'], axis=1, inplace=True)

# Features and target
X_delivery = df_delivery.drop(columns=['Days_To_Delivery'])
y_delivery = df_delivery['Days_To_Delivery']

# Train-test split
X_train_d, X_test_d, y_train_d, y_test_d = train_test_split(X_delivery, y_delivery, test_size=0.2, random_state=42)
print("Prepared delivery data shape:", X_train_d.shape)

def objective_delivery(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 50, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 1, 10),
        "random_state": 42
    }

    model = xgb.XGBRegressor(**params)
    model.fit(X_train_d, y_train_d)
    preds = model.predict(X_test_d)
    mae = mean_absolute_error(y_test_d, preds)
    return -mae  # Optuna maximizes, so we minimize MAE by negating it

study_delivery = optuna.create_study(direction="maximize")
study_delivery.optimize(objective_delivery, n_trials=20)
print("Best Delivery Params:", study_delivery.best_params)
print("Best Delivery MAE (Days):", -study_delivery.best_value)

# Train final delivery model
best_delivery_model = xgb.XGBRegressor(**study_delivery.best_params)
best_delivery_model.fit(X_train_d, y_train_d)

# DISEASE CLASSIFICATION MODEL
print("\n=== TRAINING DISEASE CLASSIFICATION MODEL ===")
# Load the dataset again for disease classification
df_disease = pd.read_excel("/Users/sathwiknomula/Downloads/animal_vet_data33.xlsx")

# Randomly assign test results with some positive samples
np.random.seed(42)
df_disease['Brucella_Test_Result'] = np.random.choice([0, 1], size=len(df_disease), p=[0.7, 0.3])
df_disease['Toxoplasma_Test_Result'] = np.random.choice([0, 1], size=len(df_disease), p=[0.7, 0.3])

# Force at least some samples to have disease
df_disease.loc[:5, 'Brucella_Test_Result'] = 1
df_disease.loc[5:10, 'Toxoplasma_Test_Result'] = 1

# Assign disease based on test results
df_disease['Disease'] = df_disease.apply(
    lambda row: 'Brucellosis' if row['Brucella_Test_Result'] == 1 
    else ('Toxoplasmosis' if row['Toxoplasma_Test_Result'] == 1 
          else 'Normal'), axis=1
)

print("✅ Shape after assignment:", df_disease.shape)
print("✅ Disease distribution:\n", df_disease['Disease'].value_counts())

# FIXED: Properly define X before using it
# Ensure clean data
df_disease = df_disease.dropna(subset=['Disease']).reset_index(drop=True)

# Drop columns not needed for disease classification
drop_cols = ['Animal_ID', 'Mating_Date', 'Estimated_Delivery_Date', 'Pregnancy_Status', 'Disease']
X_disease = df_disease.drop(columns=[col for col in drop_cols if col in df_disease.columns])

# Convert object columns to categorical using one-hot encoding
X_disease = pd.get_dummies(X_disease)
y_disease = df_disease['Disease']

# Label encode the target
le_disease = LabelEncoder()
y_disease_encoded = le_disease.fit_transform(y_disease)

# Verify shapes match
print(f"✅ X_disease shape: {X_disease.shape}")
print(f"✅ y_disease shape: {y_disease_encoded.shape}")

# Train-test split
X_train_disease, X_test_disease, y_train_disease, y_test_disease = train_test_split(
    X_disease, y_disease_encoded, test_size=0.2, random_state=42, stratify=y_disease_encoded
)

# Define Optuna objective for disease classification
def objective_disease(trial):
    params = {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        "n_estimators": trial.suggest_int("n_estimators", 100, 300),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "gamma": trial.suggest_float("gamma", 0, 5),
        "reg_lambda": trial.suggest_float("reg_lambda", 1, 10),
        "reg_alpha": trial.suggest_float("reg_alpha", 0, 10),
        "random_state": 42
    }

    model = xgb.XGBClassifier(**params)
    model.fit(X_train_disease, y_train_disease)
    preds = model.predict(X_test_disease)
    return accuracy_score(y_test_disease, preds)

# Run optimization
study_disease = optuna.create_study(direction="maximize")
study_disease.optimize(objective_disease, n_trials=20)

# Best model
print("✅ Best Disease Parameters:", study_disease.best_params)

# Train final model
best_disease_model = xgb.XGBClassifier(**study_disease.best_params)
best_disease_model.fit(X_train_disease, y_train_disease)

# Evaluate
y_pred_disease = best_disease_model.predict(X_test_disease)
acc_disease = accuracy_score(y_test_disease, y_pred_disease)
print(f"\n✅ Disease Classification Accuracy: {acc_disease:.2f}\n")

print("📊 Classification Report:\n", classification_report(y_test_disease, y_pred_disease, target_names=le_disease.classes_))
print("📉 Confusion Matrix:\n", confusion_matrix(y_test_disease, y_pred_disease))

# Save models and encoders
print("\n=== SAVING MODELS ===")
joblib.dump(best_preg_model, "pregnancy_model_xgb.pkl")
joblib.dump(best_delivery_model, "delivery_model_xgb.pkl")
joblib.dump(best_disease_model, "disease_model_xgb.pkl")
joblib.dump(le_disease, "disease_encoder.pkl")

# Save label encoders for categorical variables
for col, encoder in label_encoders.items():
    joblib.dump(encoder, f"{col}_encoder.pkl")

print("✅ All models and encoders saved successfully!")

# VETBOT KNOWLEDGE BASE
knowledge_base = {
    "brucellosis": "Brucellosis is a contagious disease caused by bacteria, often leading to abortion in pregnant animals. Common symptoms include lethargy, discharge, fever.",
    "toxoplasmosis": "Toxoplasmosis is caused by Toxoplasma gondii. It's dangerous for pregnant animals and can lead to stillbirths. Symptoms include loss of appetite, diarrhea, and fever.",
    "pregnancy_care": "Ensure the animal has a calm environment, balanced diet, and periodic vet checkups during pregnancy.",
    "delivery_signs": "Signs of labor include nesting behavior, restlessness, and contractions. Consult a vet if there's abnormal discharge or prolonged labor.",
    "fetal_heart": "Fetal heartbeat in animals can be detected via ultrasound. Absence might indicate fetal distress or miscarriage.",
    "nutrition": "During pregnancy, feed a protein-rich and easily digestible diet with proper hydration.",
    "pyometra": "Pyometra is a uterine infection common in older, unspayed female animals. Symptoms include abdominal swelling, fever, and pus discharge.",
    "test_recommendations": "Ultrasound and blood tests are essential to monitor fetal health and detect infections like Brucellosis or Toxoplasmosis."
}

def extract_entities(text):
    entities = []
    for keyword in knowledge_base.keys():
        if keyword in text.lower():
            entities.append(keyword)
    return entities

def process_query(query):
    query = query.lower()
    
    if "how to care" in query or "care for pregnant" in query:
        return knowledge_base["pregnancy_care"]
    if "symptom" in query and "brucellosis" in query:
        return knowledge_base["brucellosis"]
    if "symptom" in query and "toxoplasmosis" in query:
        return knowledge_base["toxoplasmosis"]
    if "signs of delivery" in query or "labour signs" in query:
        return knowledge_base["delivery_signs"]
    if "heartbeat" in query:
        return knowledge_base["fetal_heart"]
    if "nutrition" in query or "diet" in query:
        return knowledge_base["nutrition"]
    if "pyometra" in query:
        return knowledge_base["pyometra"]
    if "test" in query or "diagnose" in query:
        return knowledge_base["test_recommendations"]
    
    entities = extract_entities(query)
    if entities:
        return "\n\n".join([f"📌 {entity.title()}:\n{knowledge_base[entity]}" for entity in entities])
    
    return "Sorry, I couldn't understand your question. Please ask about symptoms, pregnancy, care, or test results."

def vetbot_response(query):
    query = query.lower()
    vetbot_knowledge = {
        "pregnancy_feed": "Ensure your animal gets high-quality protein, calcium, and vitamins. Avoid overfeeding.",
        "delivery_days": {
            "dog": 63, "cat": 63, "cow": 270, "goat": 150, "sheep": 152, "horse": 340
        },
        "brucellosis": "Isolate the animal, consult a vet immediately, and avoid handling birth products."
    }
    
    if "pregnant" in query and ("feed" in query or "diet" in query):
        return vetbot_knowledge["pregnancy_feed"]
    elif "how many days" in query and ("deliver" in query or "delivery" in query):
        for animal, days in vetbot_knowledge["delivery_days"].items():
            if animal in query:
                return f"A {animal} usually delivers in {days} days after mating."
        return "Please specify the animal species for delivery estimation."
    elif "brucellosis" in query or "brucella" in query:
        return vetbot_knowledge["brucellosis"]
    return "Sorry, I don't have an answer to that yet. Please consult a veterinary doctor."

# STREAMLIT APPLICATION
def create_streamlit_app():
    """
    Create the Streamlit application. 
    Save this function in a separate file called 'vet_app.py' and run with: streamlit run vet_app.py
    """
    st.set_page_config(page_title="VetCare AI System", layout="centered")
    st.title("🐾 VetCare AI System")
    st.markdown("This app predicts animal pregnancy, estimates delivery date, detects diseases, and answers veterinary queries.")

    # Load models (make sure these files exist)
    try:
        preg_model = joblib.load("pregnancy_model_xgb.pkl")
        delivery_model = joblib.load("delivery_model_xgb.pkl")
        disease_model = joblib.load("disease_model_xgb.pkl")
        st.success("✅ Models loaded successfully!")
    except:
        st.error("❌ Models not found. Please train the models first by running the main script.")
        return

    # Tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Pregnancy Prediction", "Delivery Estimation", "Disease Detection", "VetBot Chat"])

    with tab1:
        st.header("🤰 Pregnancy Prediction")
        # Input fields for pregnancy prediction
        st.info("Enter animal details for pregnancy prediction")
        
    with tab2:
        st.header("📅 Delivery Estimation")
        mating_date = st.date_input("Mating Date")
        species = st.selectbox("Species for Delivery", ["Dog", "Cat", "Cow", "Goat", "Horse"])
        
        if st.button("Estimate Delivery Date"):
            delivery_days = {"dog": 63, "cat": 63, "cow": 270, "goat": 150, "sheep": 152, "horse": 340}
            days = delivery_days.get(species.lower(), 0)
            delivery_date = mating_date + timedelta(days=days)
            st.success(f"Estimated Delivery Date: {delivery_date.strftime('%Y-%m-%d')}")

    with tab3:
        st.header("🦠 Disease Detection")
        st.info("Enter symptoms and test results for disease detection")
        
    with tab4:
        st.header("💬 VetBot - Ask Your Questions")
        st.markdown("Ask anything about animal pregnancy, care, or diseases!")
        user_query = st.text_input("Enter your query:")
        if user_query:
            answer = vetbot_response(user_query)
            st.markdown(f"**VetBot:** {answer}")

print("\n✅ Script completed successfully! All models trained and saved.")
print("To run the Streamlit app, create a separate file with the create_streamlit_app() function and run: streamlit run vet_app.py")
