import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.impute import SimpleImputer
from imblearn.over_sampling import SMOTE
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------------------------------------------
# STEP 1: LOAD DATASET
# ----------------------------------------------------
DATASET_DIR = "/content/ProjectFlask_internship_Assignment/mental_health_dataset"

FILES = {
    "Anxiety": "Anxiety.csv",
    "Depression": "Depression.csv",
    "Stress": "Stress.csv"
}

# Expected classes for each disorder
EXPECTED_CLASSES = {
    "Anxiety": ["Minimal Anxiety", "Mild Anxiety", "Moderate Anxiety", "Severe Anxiety"],
    "Depression": [
        "No Depression", "Minimal Depression", "Mild Depression",
        "Moderate Depression", "Moderately Severe Depression", "Severe Depression"
    ],
    "Stress": ["Low Stress", "Moderate Stress", "High Perceived Stress"]
}

# Expected combined labels
EXPECTED_COMBINED = [
    "Anxiety_Minimal Anxiety", "Anxiety_Mild Anxiety", "Anxiety_Moderate Anxiety", "Anxiety_Severe Anxiety",
    "Depression_No Depression", "Depression_Minimal Depression", "Depression_Mild Depression",
    "Depression_Moderate Depression", "Depression_Moderately Severe Depression", "Depression_Severe Depression",
    "Stress_Low Stress", "Stress_Moderate Stress", "Stress_High Perceived Stress"
]

# ----------------------------------------------------
# STEP 2: DATA CLEANING HELPERS
# ----------------------------------------------------
def clean_text(x: str) -> str:
    """Cleans class label text."""
    if not isinstance(x, str):
        return str(x)
    return " ".join(x.strip().title().split())  # remove extra spaces, standardize case

def unify_label(disorder: str, label: str) -> str:
    """Ensure labels match the standardized class names."""
    label = clean_text(label)

    # Define normalization rules
    replacements = {
        "High Stress": "High Perceived Stress",
        "Perceived Stress High": "High Perceived Stress",
        "Low Perceived Stress": "Low Stress",
        "Moderate  Stress": "Moderate Stress",
        "Sever Anxiety": "Severe Anxiety",
        "Moderate  Anxiety": "Moderate Anxiety",
        "Moderatly Severe Depression": "Moderately Severe Depression"
    }

    if label in replacements:
        label = replacements[label]

    # If label not found, find closest match
    if label not in EXPECTED_CLASSES[disorder]:
        for expected in EXPECTED_CLASSES[disorder]:
            if expected.lower().startswith(label.lower()[:5]) or label.lower() in expected.lower():
                label = expected
                break
        else:
            print(f"Warning: Label '{label}' not found in expected classes for {disorder}. Defaulting to {EXPECTED_CLASSES[disorder][0]}")
            label = EXPECTED_CLASSES[disorder][0]
    
    return label

# ----------------------------------------------------
# STEP 3: LOAD & PREPROCESS INDIVIDUAL CSV
# ----------------------------------------------------
def process_disorder(disorder: str, filename: str):
    path = os.path.join(DATASET_DIR, filename)
    print(f"\nProcessing {disorder} dataset...")
    df = pd.read_csv(path)
    print(f"Loaded {filename} with shape: {df.shape}")

    # Clean column names
    df.columns = [c.strip().replace(" ", "_") for c in df.columns]

    # Find label column
    label_col = next((c for c in df.columns if "label" in c.lower()), None)
    if not label_col:
        raise ValueError(f"No label column found in {filename}")

    # Debug: Print raw labels
    print(f"Raw labels for {disorder}: {df[label_col].unique()}")

    # Clean and validate labels
    df[label_col] = df[label_col].apply(lambda x: unify_label(disorder, x))
    print(f"Class distribution for {disorder}:")
    print(df[label_col].value_counts())

    # Create combined label
    df["Disorder_Type"] = disorder
    df["Combined_Label"] = disorder + "_" + df[label_col]

    # Validate combined labels
    unique_combined = df["Combined_Label"].unique()
    print(f"Combined labels for {disorder}: {unique_combined}")
    if not all(label in EXPECTED_COMBINED for label in unique_combined):
        raise ValueError(f"Unexpected combined labels in {disorder}: {unique_combined}")

    # Handle numeric columns
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    X = df[numeric_cols].copy()

    # Handle categorical columns
    categorical_cols = [
        col for col in df.columns 
        if col not in numeric_cols and col not in [label_col, "Combined_Label", "Disorder_Type"]
    ]
    for col in categorical_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))
        X[col] = df[col]

    # Handle Age
    if "1._Age" in X.columns:
        X["1._Age"] = X["1._Age"].astype(str).apply(
            lambda x: np.mean([int(i) for i in x.split('-')]) if '-' in x else pd.to_numeric(x, errors='coerce')
        )
        X["1._Age"] = X["1._Age"].fillna(X["1._Age"].mean())

    # Handle CGPA
    if "6._Current_CGPA" in X.columns:
        X["6._Current_CGPA"] = pd.to_numeric(X["6._Current_CGPA"], errors='coerce')
        if X["6._Current_CGPA"].isna().all():
            X["6._Current_CGPA"] = 3.0
        else:
            X["6._Current_CGPA"] = X["6._Current_CGPA"].fillna(X["6._Current_CGPA"].mean())

    # Impute NaNs for all numeric columns
    imputer = SimpleImputer(strategy='mean')
    X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)

    print(f"NaNs in {disorder} after preprocessing:\n{X.isna().sum()}")

    return X, df[["Combined_Label", "Disorder_Type"]], label_col

# ----------------------------------------------------
# STEP 4: COMBINE ALL DISORDERS
# ----------------------------------------------------
def combine_datasets():
    combined_X = []
    combined_y = []
    for disorder, file in FILES.items():
        X, y_df, label_col = process_disorder(disorder, file)
        combined_X.append(X)
        combined_y.append(y_df)

    # Align feature sets
    all_features = set()
    for X in combined_X:
        all_features.update(X.columns)
    all_features = sorted(list(all_features))

    aligned_X = []
    for X in combined_X:
        for col in all_features:
            if col not in X.columns:
                X[col] = 0
        aligned_X.append(X[all_features])

    final_X = pd.concat(aligned_X, ignore_index=True)
    final_y = pd.concat(combined_y, ignore_index=True)

    # Check combined label distribution
    unique_labels = sorted(final_y["Combined_Label"].unique())
    print(f"\n✅ Combined dataset shape: {final_X.shape}")
    print(f"Unique Combined Labels ({len(unique_labels)}): {unique_labels}")

    # Verify against EXPECTED_COMBINED
    unexpected_labels = [lbl for lbl in unique_labels if lbl not in EXPECTED_COMBINED]
    if unexpected_labels:
        print(f"⚠️ Found unexpected labels: {unexpected_labels}")
    else:
        print("✅ All combined labels match expected list.")

    # Dynamically handle number of classes (don’t raise hardcoded error)
    print(f"\nDetected {len(unique_labels)} unique target classes.")

    return final_X, final_y


# ----------------------------------------------------
# STEP 5: TRAIN MODEL
# ----------------------------------------------------
def train_combined_model(X, y):
    # Encode labels
    le = LabelEncoder()
    y_encoded = le.fit_transform(y["Combined_Label"])

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # SMOTE balancing
    try:
        smote = SMOTE(random_state=42, k_neighbors=3)
        X_resampled, y_resampled = smote.fit_resample(X_scaled, y_encoded)
        print(f"✅ Applied SMOTE: {X.shape} → {X_resampled.shape}")
    except Exception as e:
        print(f"⚠️ SMOTE skipped: {e}")
        X_resampled, y_resampled = X_scaled, y_encoded

    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_resampled, y_resampled, test_size=0.2, random_state=42, stratify=y_resampled
    )

    # Model with hyperparameter tuning
    model = RandomForestClassifier(random_state=42, class_weight="balanced")
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }

    # Grid Search
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(model, param_grid, cv=cv, scoring='f1_weighted', n_jobs=-1)
    print("\n🚀 Running GridSearchCV...")
    grid_search.fit(X_train, y_train)
    model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")

    # Cross-validation
    cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring='f1_weighted')
    print(f"\n🚀 Cross-validation F1 scores: {cv_scores}")
    print(f"Average CV F1 score: {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

    print("\n🚀 Training final model...")
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"✅ Test Accuracy: {acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print("\nConfusion Matrix:\n", cm)

    # Plot confusion matrix
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=le.classes_, yticklabels=le.classes_)
    plt.title('Confusion Matrix for Combined Dataset')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    # Feature importance
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': model.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    print(f"\nFeature Importance:")
    print(feature_importance)

    plt.figure(figsize=(10, 6))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance for Combined Dataset')
    plt.tight_layout()
    plt.show()

    # Save
    joblib.dump(model, "mental_health_combined_model.pkl")
    joblib.dump(le, "label_encoder.pkl")
    joblib.dump(scaler, "scaler.pkl")
    print("\n💾 Model and encoders saved!")

    return model, le, scaler

# ----------------------------------------------------
# STEP 6: MAIN
# ----------------------------------------------------
if __name__ == "__main__":
    X, y = combine_datasets()
    model, le, scaler = train_combined_model(X, y)
    print("\n🎯 Done! Clean, unified model trained successfully.")
