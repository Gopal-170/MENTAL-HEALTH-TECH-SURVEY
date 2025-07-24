# === Imports ===
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import matplotlib.pyplot as plt
import joblib

# === Load data ===
df = pd.read_csv('survey.csv')

# === Cleaning ===

# Remove age outliers
df = df[(df['Age'] >= 18) & (df['Age'] <= 75)]

# Clean gender
def clean_gender(g):
    g = str(g).strip().lower()
    male = ['male', 'm', 'man', 'msle', 'mal', 'maile', 'male-ish', 'cis male', 'cis man', 'make', 'malr', 'mail']
    female = ['female', 'f', 'woman', 'cis female', 'femake', 'femail', 'female ', 'female (cis)']
    if any(word in g for word in male):
        return 'Male'
    elif any(word in g for word in female):
        return 'Female'
    else:
        return 'Other'
df['Gender'] = df['Gender'].apply(clean_gender)

# Fill missing
df['self_employed'] = df['self_employed'].fillna('Unknown')
df['work_interfere'] = df['work_interfere'].fillna('Unknown')

# Drop unused columns
df = df.drop(['Timestamp', 'comments'], axis=1)

# Age groups
bins = [18, 25, 35, 45, 60, 75]
labels = ['18–25', '26–35', '36–45', '46–60', '61–75']
df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels, include_lowest=True)
df['age_group'] = df['age_group'].astype(str)
df = df.drop('Age', axis=1)

# === Features & Target ===
X = df.drop('treatment', axis=1)
y = df['treatment'].map({'Yes': 1, 'No': 0})

# === Train-Test Split ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === Categorical Columns Detection ===
categorical_cols = X.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()

# === Encoding Pipeline (with Imputation) ===
cat_pipeline = make_pipeline(
    SimpleImputer(strategy='most_frequent'),
    OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1)
)

encoder = ColumnTransformer(
    transformers=[
        ('cat', cat_pipeline, categorical_cols)
    ],
    remainder='passthrough'
)

# === Pipelines ===
lr_pipeline = Pipeline(steps=[
    ('preprocessor', encoder),
    ('classifier', LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline(steps=[
    ('preprocessor', encoder),
    ('classifier', RandomForestClassifier(random_state=42))
])

# === Fit Models ===
lr_pipeline.fit(X_train, y_train)
rf_pipeline.fit(X_train, y_train)

# === Predictions ===
lr_pred = lr_pipeline.predict(X_test)
rf_pred = rf_pipeline.predict(X_test)

# === Evaluation Function ===
def evaluate(name, y_true, y_pred):
    print(f"\n🔹 {name} Evaluation")
    print("Confusion Matrix:\n", confusion_matrix(y_true, y_pred))
    print("Classification Report:\n", classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))

# === Evaluate Both ===
evaluate("Logistic Regression", y_test, lr_pred)
evaluate("Random Forest", y_test, rf_pred)

# === Feature Importance for RF ===
# Only valid if ColumnTransformer did not reorder/drop any
print("\n🔹 Random Forest Feature Importance Plot:")
rf_model = rf_pipeline.named_steps['classifier']
encoded_features = categorical_cols + list(X.select_dtypes(include=[np.number]).columns)

importances = rf_model.feature_importances_
feat_series = pd.Series(importances, index=encoded_features).sort_values(ascending=True)

plt.figure(figsize=(8, 6))
feat_series.tail(15).plot(kind='barh')
plt.title("Top 15 Feature Importances (Random Forest)")
plt.tight_layout()
plt.show()
