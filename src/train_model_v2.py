"""
RETRAINED MODEL - Version 2
Only derivable features from 11 form inputs
Removed non-derivable features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
from pathlib import Path

# ============================================================
# STEP 1 — Load the Dataset
# ============================================================
print("="*60)
print("STEP 1 — Load and Prepare Dataset")
print("="*60)

dataset_path = Path("data/processed/insurance_claims_feature_engineered.csv")
df = pd.read_csv(dataset_path)

print(f"Dataset shape: {df.shape}")
print(f"Fraud rate: {df['fraud_reported'].mean()*100:.2f}%")

# ============================================================
# STEP 2 — Select ONLY Derivable Features
# ============================================================
print("\n" + "="*60)
print("STEP 2 — Select Derivable Features Only")
print("="*60)

# Features derivable from 11 form inputs:
# 1. age
# 2. insured_sex (Gender)
# 3. policy_state (Region)
# 4. auto_year (Vehicle Age)
# 5. total_claim_amount (Claim Amount)
# 6. months_as_customer (Claim Frequency)
# 7. incident_severity (derived)
# 8. bodily_injuries (Injury Severity)
# 9. collision_type (NEW - Collision Type input)
# 10. injury_severity_score (NEW - Injury Severity input)
# 11. claim_to_premium_ratio, high_claim_indicator, vehicle_damage_flag (key derived)
# 12. claim_amount_category (derived from claim amount)
# 13. claim_to_vehicle_ratio (derived)

DERIVABLE_FEATURES = [
    'age',
    'insured_sex',
    'policy_state',
    'auto_year',
    'total_claim_amount',
    'months_as_customer',
    'incident_severity',
    'bodily_injuries',
    'collision_type',
    'injury_severity_score',
    'claim_to_premium_ratio',
    'high_claim_indicator',
    'vehicle_damage_flag',
    'claim_amount_category',
    'claim_to_vehicle_ratio',
    'incident_hour_of_the_day',
    'incident_type'
]

# Check which features exist in the dataset
available_features = [f for f in DERIVABLE_FEATURES if f in df.columns]
missing_features = [f for f in DERIVABLE_FEATURES if f not in df.columns]

print(f"Available derivable features: {len(available_features)}")
print(f"Missing features: {missing_features}")
print(f"\nUsing features:")
for i, feat in enumerate(available_features, 1):
    print(f"  {i:2d}. {feat}")

# ============================================================
# STEP 3 — Prepare Features and Target
# ============================================================
print("\n" + "="*60)
print("STEP 3 — Prepare Features and Target")
print("="*60)

X = df[available_features].copy()
y = df['fraud_reported'].copy()

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Target distribution: {y.value_counts().to_dict()}")

# ============================================================
# STEP 4 — Identify and Encode Categorical Features
# ============================================================
print("\n" + "="*60)
print("STEP 4 — Identify and Encode Categorical Features")
print("="*60)

categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

print(f"Categorical columns: {categorical_cols}")
print(f"Numerical columns: {len(numerical_cols)}")

# Handle missing values only for numerical columns
for col in numerical_cols:
    X[col] = X[col].fillna(X[col].mean())

# One-hot encode categorical features
X = pd.get_dummies(X, columns=categorical_cols, drop_first=True, dtype=int)

print(f"Features after encoding: {X.shape}")
print(f"Total features for model: {X.shape[1]}")

# ============================================================
# STEP 5 — Train/Test Split and Scaling
# ============================================================
print("\n" + "="*60)
print("STEP 5 — Train/Test Split and Feature Scaling")
print("="*60)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Training set: {X_train.shape[0]} samples")
print(f"Test set: {X_test.shape[0]} samples")
print(f"Training fraud rate: {y_train.mean()*100:.2f}%")
print(f"Test fraud rate: {y_test.mean()*100:.2f}%")

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("StandardScaler fit on training data and applied to train and test.")

# ============================================================
# STEP 6 — Configure XGBoost with Class Imbalance
# ============================================================
print("\n" + "="*60)
print("STEP 6 — Configure XGBoost and Class Imbalance")
print("="*60)

# Calculate scale_pos_weight for imbalanced data
n_legitimate = (y_train == 0).sum()
n_fraud = (y_train == 1).sum()
scale_pos_weight = n_legitimate / n_fraud

print(f"Training set: {n_legitimate} legitimate, {n_fraud} fraud")
print(f"scale_pos_weight: {scale_pos_weight:.4f}")

# Configure with aggressive regularization for small feature set
model = XGBClassifier(
    objective='binary:logistic',
    eval_metric='logloss',
    learning_rate=0.05,
    max_depth=4,
    min_child_weight=8,
    subsample=0.7,
    colsample_bytree=0.7,
    reg_alpha=1.5,
    reg_lambda=2.0,
    n_estimators=120,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    tree_method='hist',
    n_jobs=-1
)

print("\nXGBoost configured with:")
print(f"  • learning_rate=0.05")
print(f"  • max_depth=4")
print(f"  • min_child_weight=8")
print(f"  • subsample=0.7, colsample_bytree=0.7")
print(f"  • reg_alpha=1.5, reg_lambda=2.0")
print(f"  • n_estimators=120")

# ============================================================
# STEP 7 — Train the Model
# ============================================================
print("\n" + "="*60)
print("STEP 7 — Train the Model")
print("="*60)

model.fit(X_train_scaled, y_train)
print("Model training complete.")

# ============================================================
# STEP 8 — Evaluate the Model
# ============================================================
print("\n" + "="*60)
print("STEP 8 — Evaluate the Model")
print("="*60)

y_train_pred = model.predict(X_train_scaled)
y_test_pred = model.predict(X_test_scaled)

print("\nTRAINING SET:")
print(f"  Accuracy:  {accuracy_score(y_train, y_train_pred):.4f}")
print(f"  Precision: {precision_score(y_train, y_train_pred):.4f}")
print(f"  Recall:    {recall_score(y_train, y_train_pred):.4f}")
print(f"  F1 Score:  {f1_score(y_train, y_train_pred):.4f}")

print("\nTEST SET:")
test_accuracy = accuracy_score(y_test, y_test_pred)
test_precision = precision_score(y_test, y_test_pred)
test_recall = recall_score(y_test, y_test_pred)
test_f1 = f1_score(y_test, y_test_pred)

print(f"  Accuracy:  {test_accuracy:.4f}")
print(f"  Precision: {test_precision:.4f}")
print(f"  Recall:    {test_recall:.4f}")
print(f"  F1 Score:  {test_f1:.4f}")

print("\nConfusion matrix (rows=true, cols=predicted):")
cm = confusion_matrix(y_test, y_test_pred)
print(f"              Predicted 0   Predicted 1")
print(f"  Actual 0: {cm[0,0]:13d} {cm[0,1]:13d}   (Legitimate)")
print(f"  Actual 1: {cm[1,0]:13d} {cm[1,1]:13d}   (Fraud)")

# ============================================================
# STEP 9 — Feature Importance
# ============================================================
print("\n" + "="*60)
print("STEP 9 — Feature Importance")
print("="*60)

feature_importance = pd.DataFrame({
    'feature': X.columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:40s}: {row['importance']:.4f}")

# ============================================================
# STEP 10 — Save Model Artifacts
# ============================================================
print("\n" + "="*60)
print("STEP 10 — Save Model Artifacts (V2)")
print("="*60)

models_dir = Path("models")
models_dir.mkdir(exist_ok=True)

# Save with v2 suffix to differentiate
model_path = models_dir / "fraud_model_v2.pkl"
scaler_path = models_dir / "scaler_v2.pkl"
features_path = models_dir / "feature_columns_v2.pkl"

joblib.dump(model, model_path)
joblib.dump(scaler, scaler_path)
joblib.dump(X.columns.tolist(), features_path)

print(f"Saved: {model_path}")
print(f"Saved: {scaler_path}")
print(f"Saved: {features_path}")

# ============================================================
# STEP 11 — Comparison Summary
# ============================================================
print("\n" + "="*60)
print("STEP 11 — Model Comparison (V1 vs V2)")
print("="*60)

comparison = f"""
╔════════════════════════════════════════════════════════════╗
║          MODEL V1 (ALL FEATURES) vs V2 (DERIVABLE)        ║
╠════════════════════════════════════════════════════════════╣
║ Metric                 V1 (Old)      V2 (New)   Change     ║
╠════════════════════════════════════════════════════════════╣
║ Features               46            {X.shape[1]:2d}           {X.shape[1]-46:+d}         ║
║ F1 Score               0.7304        {test_f1:.4f}   {test_f1-0.7304:+.4f}     ║
║ Recall                 0.8571        {test_recall:.4f}   {test_recall-0.8571:+.4f}     ║
║ Precision              0.6364        {test_precision:.4f}   {test_precision-0.6364:+.4f}     ║
║ Accuracy               0.8450        {test_accuracy:.4f}   {test_accuracy-0.8450:+.4f}     ║
╠════════════════════════════════════════════════════════════╣
║ Key Insight:                                               ║
║ V2 uses ONLY derivable features from 11 form inputs       ║
║ No guessing or hallucination of features                  ║
╚════════════════════════════════════════════════════════════╝
"""

print(comparison)

print("\nModel V2 trained and saved successfully!")
print("Files saved with _v2 suffix for comparison.")
