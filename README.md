# Down-Syndrome-predictor
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

# Likelihood Ratios
LRs = {
    'MaternalAge': {35: 1.0, 40: 3.5, 45: 11.7},
    'PaternalAge': {35: 1.0, 40: 3.0, 45: 5.5},
    'MotherCarrier': 10,

    'FatherCarrier': 3,
    'MotherKaryotype': 10,
    'FatherKaryotype': 3,
    'SNPAbnormal': 2
}

def plot_model_evaluation(model, X, y, dataset_name=""):
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {dataset_name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y, y_prob)
    auc_score = roc_auc_score(y, y_prob)

    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.2f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random Model")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {dataset_name}")
    plt.legend()
    plt.grid(True)
    plt.show()

    print(f"AUC Score: {auc_score:.2f}")

def get_prior_odds(age):
    if age >= 45:
        risk = 1/30
    elif age >= 40:
        risk = 1/100
    elif age >= 35:
        risk = 1/350
    else:
        risk = 1/1250
    return risk / (1 - risk)

def calculate_posterior_probability(age, paternal_age, mc, fc, mk, fk, snp):
    prior_odds = get_prior_odds(age)
    total_LR = 1
    if mc: total_LR *= LRs['MotherCarrier']
    if fc: total_LR *= LRs['FatherCarrier']
    if mk: total_LR *= LRs['MotherKaryotype']
    if fk: total_LR *= LRs['FatherKaryotype']
    if snp: total_LR *= LRs['SNPAbnormal']
    if paternal_age >= 45:
        total_LR *= LRs['PaternalAge'][45]
    elif paternal_age >= 40:
        total_LR *= LRs['PaternalAge'][40]
    elif paternal_age >= 35:
        total_LR *= LRs['PaternalAge'][35]
    posterior_odds = prior_odds * total_LR
    return posterior_odds / (1 + posterior_odds)

# Data Preprocessing
def preprocess_data(df):
    rename_map = {
        'age': 'MaternalAge',
        'maternal_age': 'MaternalAge',
        'paternal_age': 'PaternalAge',
        'mother_carrier_status': 'MotherCarrier',
        'father_carrier_status': 'FatherCarrier',
        'mother_karyotype': 'MotherKaryotype',
        'father_karyotype': 'FatherKaryotype',
        'snp_abnormal': 'SNP',
        'diagnosis': 'DS_Risk',
        'has_down_syndrome': 'DS_Risk',
        'ds_risk': 'DS_Risk'
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    df = df.dropna()
    binary_cols = ["MotherCarrier", "FatherCarrier", "MotherKaryotype", "FatherKaryotype", "SNP", "DS_Risk"]
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].astype(int).clip(0, 1)
    df["MaternalAge"] = pd.to_numeric(df["MaternalAge"], errors="coerce")
    df["PaternalAge"] = pd.to_numeric(df["PaternalAge"], errors="coerce")
    df = df[df["MaternalAge"].between(15, 50)]
    df = df[df["PaternalAge"].between(15, 60)]
    df["BayesProb"] = df.apply(lambda row: calculate_posterior_probability(
        row["MaternalAge"], row["PaternalAge"],
        row["MotherCarrier"], row["FatherCarrier"],
        row["MotherKaryotype"], row["FatherKaryotype"],
        row["SNP"]
    ), axis=1)
    return df.reset_index(drop=True)

# Load and Clean Data
df = pd.read_excel("paste address of ur saved file which has the data")
df_clean = preprocess_data(df)

# Features and Target
X = df_clean.drop("DS_Risk", axis=1)
y = df_clean["DS_Risk"]

# Split Data
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=1/3, random_state=42)

# Apply SMOTE
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Standardize
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_res)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Train Logistic Regression
log_model = LogisticRegression(max_iter=1000, class_weight='balanced')
log_model.fit(X_train_scaled, y_train_res)

# Train Neural Network (MLP)
mlp_model = MLPClassifier(hidden_layer_sizes=(20, 10), max_iter=1000, random_state=42)
mlp_model.fit(X_train_scaled, y_train_res)

# Evaluate Models
print("\n--- Logistic Regression ---")
print("Validation:\n", classification_report(y_val, log_model.predict(X_val_scaled), zero_division=1))
print("Test:\n", classification_report(y_test, log_model.predict(X_test_scaled), zero_division=1))
plot_model_evaluation(log_model, X_test_scaled, y_test, dataset_name="Logistic Regression")

print("\n--- Neural Network (MLP) ---")
print("Validation:\n", classification_report(y_val, mlp_model.predict(X_val_scaled), zero_division=1))
print("Test:\n", classification_report(y_test, mlp_model.predict(X_test_scaled), zero_division=1))
plot_model_evaluation(mlp_model, X_test_scaled, y_test, dataset_name="Neural Network (MLP)")

# Interactive Prediction
def run_console_prediction():
    print("\nEnter patient information for DS risk prediction:")
    age = int(input("Maternal Age: "))
    paternal_age = int(input("Paternal Age: "))
    mc = int(input("Mother is Carrier (0/1): "))
    fc = int(input("Father is Carrier (0/1): "))
    mk = int(input("Mother Karyotype Abnormal (0/1): "))
    fk = int(input("Father Karyotype Abnormal (0/1): "))
    snp = int(input("SNP Abnormality Detected (0/1): "))

    bayes_prob = calculate_posterior_probability(age, paternal_age, mc, fc, mk, fk, snp)
    print(f"\n[Bayesian Estimate] Risk of Down Syndrome: {bayes_prob*100:.2f}%")

    input_df = pd.DataFrame([{
        "MaternalAge": age, "PaternalAge": paternal_age,
        "MotherCarrier": mc, "FatherCarrier": fc,
        "MotherKaryotype": mk, "FatherKaryotype": fk,
        "SNP": snp, "BayesProb": bayes_prob
    }])

    # Ensure column order matches the training data
    input_df = input_df[X_train.columns]

    input_scaled = scaler.transform(input_df)
    prob_log = log_model.predict_proba(input_scaled)[0][1]
    prob_mlp = mlp_model.predict_proba(input_scaled)[0][1]

    print(f"[Logistic Model Estimate] Risk: {prob_log*100:.2f}%")
    print(f"[Neural Net Estimate] Risk: {prob_mlp*100:.2f}%")

# Run prediction
run_console_prediction()
