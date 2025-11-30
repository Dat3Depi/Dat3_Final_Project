
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================
# 1️⃣ Load Dataset
# ============================================================

file_path = "Hr_Data.xlsx"
sheets = pd.read_excel(file_path, sheet_name=None)
print("Sheets found:", list(sheets.keys()))

# Load specific sheets
employees = sheets["Employees"]
performance = sheets["PerformanceRating"]

# Merge by EmployeeID
if "EmployeeID" in employees.columns and "EmployeeID" in performance.columns:
    df = pd.merge(employees, performance, on="EmployeeID", how="left")
else:
    raise ValueError("Both sheets must contain 'EmployeeID' to merge")

print("Merged dataset shape:", df.shape)

# ============================================================
# 2️⃣ Data Cleaning & Conversion
# ============================================================

df['OverTime_flag'] = df['OverTime'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)
df['Attrition_flag'] = df['Attrition'].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0).astype(int)

# BusinessTravel mapping (based on your categories)
df['BusinessTravel_flag'] = df['BusinessTravel'].astype(str).map({
    'Frequent Traveller': 1,
    'Some Travel': 0.5,
    'No Travel': 0
}).fillna(0)

# Convert numeric columns safely
numeric_cols = [
    'DistanceFromHome (KM)', 'YearsWithCurrManager', 'StockOptionLevel',
    'YearsSinceLastPromotion', 'Salary', 'Age', 'YearsAtCompany', 'JobSatisfaction',
    'WorkLifeBalance', 'EnvironmentSatisfaction', 'TrainingOpportunitiesWithinYear'
]
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

# ============================================================
# 3️⃣ Create Burnout Index & Loyalty Score
# ============================================================

def normalize(series):
    if series.max() == 0:
        return series.fillna(0)
    return series / series.max()

# Burnout Index
df['Distance_norm'] = normalize(df['DistanceFromHome (KM)'])
df['BurnoutIndex'] = (
    (df['OverTime_flag'] * 0.4) +
    (df['BusinessTravel_flag'] * 0.3) +
    (df['Distance_norm'] * 0.3)
)

# Loyalty Score
df['YearsWithManager_norm'] = normalize(df['YearsWithCurrManager'])
df['StockOption_norm'] = normalize(df['StockOptionLevel'])
df['LoyaltyScore'] = (
    (df['YearsWithManager_norm'] * 0.5) +
    (df['StockOption_norm'] * 0.3) +
    ((1 / (1 + df['YearsSinceLastPromotion'])) * 0.2)
)


features = [
    'YearsSinceLastPromotion',
    'BurnoutIndex',
    'LoyaltyScore',
    'JobSatisfaction',
    'Salary',
    'YearsWithCurrManager',
    'WorkLifeBalance',
    'EnvironmentSatisfaction',
    'TrainingOpportunitiesWithinYear'
]

X = df[features].fillna(0)
y = df['Attrition_flag']

# --- Balance the dataset using SMOTE ---
print("\n Balancing dataset using SMOTE...")
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)
print("After SMOTE: ", X_resampled.shape, "rows")

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
)

# --- Tuned Random Forest Model ---
model = RandomForestClassifier(
    n_estimators=300,
    max_depth=10,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    class_weight='balanced'
)
model.fit(X_train, y_train)

# ============================================================
# 6️⃣ Evaluate Model
# ============================================================

y_pred = model.predict(X_test)
acc = round(accuracy_score(y_test, y_pred) * 100, 2)
print(f"\n Model Accuracy: {acc}%")
print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

# Confusion Matrix Visualization
plt.figure(figsize=(5, 4))
sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix - HR Attrition Model")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# ============================================================
# 7️⃣ Feature Importance Visualization
# ============================================================

importances = model.feature_importances_
feat_imp = pd.Series(importances, index=features).sort_values(ascending=True)

plt.figure(figsize=(8, 5))
feat_imp.plot(kind='barh', color='teal')
plt.title("Feature Importance - HR Attrition Model")
plt.xlabel("Importance Score")
plt.ylabel("Feature")
plt.tight_layout()
plt.show()

# ============================================================
# 8️⃣ Predict Attrition Risk for All Employees
# ============================================================

# ============================================================
# 🔟 Predict for All Employees & Improved Risk Classification
# ============================================================

df['PredictedRiskProb'] = model.predict_proba(X)[:, 1]

# Better, data-driven bins for attrition probability
risk_bins = [0.0, 0.30, 0.55, 0.80, 1.0]
risk_labels = ['Low', 'Moderate', 'High', 'Critical']

df['PredictedRisk'] = pd.cut(
    df['PredictedRiskProb'],
    bins=risk_bins,
    labels=risk_labels,
    include_lowest=True
)

# HR Alert flag
df['HR_Alert'] = np.where(df['PredictedRisk'].isin(['High', 'Critical']), 'Action Required', 'Stable')

df = df.round(2)

# Save results
output_file = "HR_Predictions.xlsx"
df.to_excel(output_file, index=False)

# Preview
print("\n Sample Output:")
print(df[['EmployeeID','BurnoutIndex','LoyaltyScore','PredictedRisk','PredictedRiskProb','HR_Alert']].head(10))
