# %% ============================================================
# CELL 1 - IMPORT LIBRARIES
# ============================================================

import os
import random
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
    cross_val_score
)

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
    roc_curve
)

from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

from xgboost import XGBClassifier

print("Libraries imported successfully.")


# %% ============================================================
# CELL 3 - SET RANDOM SEED
# ============================================================

SEED=42

random.seed(SEED)
np.random.seed(SEED)

print("Random Seed set to:", SEED)

# %% ============================================================
# CELL 4 - LOAD DATASET
# ============================================================

column_names = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
    "Outcome"
]

local_file="diabetes.csv"
dataset_url=("https://raw.githubusercontent.com/"
    "jbrownlee/Datasets/master/"
    "pima-indians-diabetes.data.csv"
    )

if os.path.exists(local_file):
    print("Loading local dataset:", local_file)

    df = pd.read_csv(local_file)

    # Handle datasets without correct column names
    if "Outcome" not in df.columns:
        df = pd.read_csv(
            local_file,
            names=column_names
        )

else:
    print("Local dataset not found.")
    print("Downloading dataset...")

    df = pd.read_csv(
        dataset_url,
        names=column_names
    )

print("\nDataset loaded successfully.")
print("Dataset shape:", df.shape)
# %%
# %% ============================================================
# CELL 5 - DISPLAY DATASET
# ============================================================

print("First 10 rows:")
print(df.head(10))
# %%
# %% ============================================================
# CELL 6 - DATASET INFORMATION
# ============================================================

print("Dataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())

print("\nDataset information:")
df.info()

print("\nStatistical summary:")
print(df.describe().T)
# %%
# %% ============================================================
# CELL 7 - CHECK MISSING VALUES
# ============================================================

print("Missing values:")
print(df.isnull().sum())
# %%
# %% ============================================================
# CELL 8 - CHECK PHYSIOLOGICALLY INVALID ZERO VALUES
# ============================================================

zero_invalid_columns = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

print("Zero values in clinical features:\n")

for column in zero_invalid_columns:
    count = (df[column] == 0).sum()

    print(
        f"{column:25s}: {count}"
    )
# %%
# %% ============================================================
# CELL 9 - REPLACE INVALID ZEROS WITH NaN
# ============================================================

df_clean = df.copy()

df_clean[zero_invalid_columns] = (
    df_clean[zero_invalid_columns]
    .replace(0, np.nan)
)

print("Missing values after replacing invalid zeros:\n")
print(df_clean.isnull().sum())
# %%
# %% ============================================================
# CELL 10 - CLASS DISTRIBUTION
# ============================================================

class_counts = (
    df_clean["Outcome"]
    .value_counts()
    .sort_index()
)

print("Class distribution:")
print(class_counts)

print("\nClass percentages:")

print(
    df_clean["Outcome"]
    .value_counts(normalize=True)
    .sort_index()
    * 100
)
# %%
# %% ============================================================
# CELL 11 - VISUALIZE CLASS IMBALANCE
# ============================================================

labels = [
    "No Diabetes (0)",
    "Diabetes (1)"
]

values = [
    class_counts.get(0, 0),
    class_counts.get(1, 0)
]

plt.figure(figsize=(7, 5))

plt.bar(
    labels,
    values
)

plt.title("Original Diabetes Class Distribution")
plt.ylabel("Number of Patients")

for i, value in enumerate(values):
    plt.text(
        i,
        value + 5,
        str(value),
        ha="center"
    )

plt.tight_layout()
plt.show()
# %%
# %% ============================================================
# CELL 12 - FEATURES AND TARGET
# ============================================================

X = df_clean.drop(
    columns=["Outcome"]
)

y = df_clean["Outcome"]

print("Feature shape:", X.shape)
print("Target shape:", y.shape)

print("\nFeatures:")
print(X.columns.tolist())
# %%
# %% ============================================================
# CELL 13 - TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=SEED,
    stratify=y
)

print("Training samples:", len(X_train))
print("Testing samples :", len(X_test))

print("\nTraining class distribution:")
print(y_train.value_counts())

print("\nTesting class distribution:")
print(y_test.value_counts())
# %%
# %% ============================================================
# CELL 14 - STRATIFIED CROSS VALIDATION
# ============================================================

cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=SEED
)

print("Cross-validation strategy:")
print(cv)
# %%
# %% ============================================================
# CELL 15 - DEMONSTRATE SMOTE
# ============================================================

demo_imputer = SimpleImputer(
    strategy="median"
)

X_train_demo = demo_imputer.fit_transform(
    X_train
)

demo_smote = SMOTE(
    random_state=SEED
)

X_demo_resampled, y_demo_resampled = (
    demo_smote.fit_resample(
        X_train_demo,
        y_train
    )
)

print("Before SMOTE:")
print(y_train.value_counts())

print("\nAfter SMOTE:")
print(
    pd.Series(
        y_demo_resampled
    ).value_counts()
)
# %%
# %% ============================================================
# CELL 16 - VISUALIZE SMOTE EFFECT
# ============================================================

before_counts = y_train.value_counts().sort_index()

after_counts = (
    pd.Series(y_demo_resampled)
    .value_counts()
    .sort_index()
)

x = np.arange(2)
width = 0.35

plt.figure(figsize=(8, 5))

plt.bar(
    x - width / 2,
    before_counts.values,
    width,
    label="Before SMOTE"
)

plt.bar(
    x + width / 2,
    after_counts.values,
    width,
    label="After SMOTE"
)

plt.xticks(
    x,
    ["No Diabetes", "Diabetes"]
)

plt.ylabel("Number of Samples")
plt.title("Training Distribution Before and After SMOTE")
plt.legend()

plt.tight_layout()
plt.show()
# %%
# %% ============================================================
# CELL 17 - SVM PIPELINE
# ============================================================

svm_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "scaler",
            StandardScaler()
        ),

        (
            "smote",
            SMOTE(
                random_state=SEED
            )
        ),

        (
            "classifier",
            SVC(
                probability=True,
                random_state=SEED
            )
        )
    ]
)

print(svm_pipeline)
# %%
# %% ============================================================
# CELL 18 - SVM HYPERPARAMETERS
# ============================================================

svm_param_grid = {

    "classifier__kernel": [
        "rbf"
    ],

    "classifier__C": [
        0.5,
        1,
        5,
        10
    ],

    "classifier__gamma": [
        "scale",
        0.01,
        0.1
    ]
}

print("SVM parameter combinations prepared.")
# %%
# %% ============================================================
# CELL 19 - SVM GRID SEARCH
# ============================================================

svm_grid = GridSearchCV(
    estimator=svm_pipeline,
    param_grid=svm_param_grid,
    scoring="recall",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

svm_grid.fit(
    X_train,
    y_train
)

print("\nBest SVM Parameters:")
print(svm_grid.best_params_)

print("\nBest SVM CV Recall:")
print(
    round(
        svm_grid.best_score_,
        4
    )
)
# %%
# %% ============================================================
# CELL 20 - RANDOM FOREST PIPELINE
# ============================================================

rf_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "smote",
            SMOTE(
                random_state=SEED
            )
        ),

        (
            "classifier",
            RandomForestClassifier(
                random_state=SEED,
                n_jobs=-1
            )
        )
    ]
)

print(rf_pipeline)
# %%
# %% ============================================================
# CELL 21 - RANDOM FOREST HYPERPARAMETERS
# ============================================================

rf_param_grid = {

    "classifier__n_estimators": [
        200,
        400
    ],

    "classifier__max_depth": [
        None,
        6,
        10
    ],

    "classifier__min_samples_split": [
        2,
        5
    ],

    "classifier__min_samples_leaf": [
        1,
        2
    ],

    "classifier__max_features": [
        "sqrt"
    ]
}

print("Random Forest parameter grid prepared.")
# %%
# %% ============================================================
# CELL 22 - RANDOM FOREST GRID SEARCH
# ============================================================

rf_grid = GridSearchCV(
    estimator=rf_pipeline,
    param_grid=rf_param_grid,
    scoring="recall",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

rf_grid.fit(
    X_train,
    y_train
)

print("\nBest Random Forest Parameters:")
print(rf_grid.best_params_)

print("\nBest Random Forest CV Recall:")
print(
    round(
        rf_grid.best_score_,
        4
    )
)
# %%
# %% ============================================================
# CELL 23 - XGBOOST PIPELINE
# ============================================================

xgb_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        ),

        (
            "smote",
            SMOTE(
                random_state=SEED
            )
        ),

        (
            "classifier",
            XGBClassifier(
                random_state=SEED,
                eval_metric="logloss",
                n_jobs=-1
            )
        )
    ]
)

print(xgb_pipeline)
# %%
# %% ============================================================
# CELL 24 - XGBOOST HYPERPARAMETERS
# ============================================================

xgb_param_grid = {

    "classifier__n_estimators": [
        150,
        250
    ],

    "classifier__max_depth": [
        3,
        5
    ],

    "classifier__learning_rate": [
        0.03,
        0.1
    ],

    "classifier__subsample": [
        0.8,
        1.0
    ],

    "classifier__colsample_bytree": [
        0.8
    ]
}

print("XGBoost parameter grid prepared.")
# %%
# %% ============================================================
# CELL 25 - XGBOOST GRID SEARCH
# ============================================================

xgb_grid = GridSearchCV(
    estimator=xgb_pipeline,
    param_grid=xgb_param_grid,
    scoring="recall",
    cv=cv,
    n_jobs=-1,
    verbose=1,
    return_train_score=True
)

xgb_grid.fit(
    X_train,
    y_train
)

print("\nBest XGBoost Parameters:")
print(xgb_grid.best_params_)

print("\nBest XGBoost CV Recall:")
print(
    round(
        xgb_grid.best_score_,
        4
    )
)
# %%
# %% ============================================================
# CELL 26 - BEST MODELS
# ============================================================

best_svm = svm_grid.best_estimator_
best_rf = rf_grid.best_estimator_
best_xgb = xgb_grid.best_estimator_

print("Best models extracted successfully.")
# %%
# %% ============================================================
# CELL 27 - CROSS-VALIDATION RECALL ARRAYS
# ============================================================

svm_cv_recall = cross_val_score(
    best_svm,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

rf_cv_recall = cross_val_score(
    best_rf,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

xgb_cv_recall = cross_val_score(
    best_xgb,
    X_train,
    y_train,
    cv=cv,
    scoring="recall",
    n_jobs=-1
)

print("SVM Recall CV:")
print(
    np.round(
        svm_cv_recall,
        4
    )
)

print("\nRandom Forest Recall CV:")
print(
    np.round(
        rf_cv_recall,
        4
    )
)

print("\nXGBoost Recall CV:")
print(
    np.round(
        xgb_cv_recall,
        4
    )
)
# %%
# %% ============================================================
# CELL 28 - CV RECALL SUMMARY
# ============================================================

print("SVM")
print(
    "Mean Recall:",
    round(
        svm_cv_recall.mean(),
        4
    )
)

print(
    "Std:",
    round(
        svm_cv_recall.std(),
        4
    )
)

print("\nRandom Forest")
print(
    "Mean Recall:",
    round(
        rf_cv_recall.mean(),
        4
    )
)

print(
    "Std:",
    round(
        rf_cv_recall.std(),
        4
    )
)

print("\nXGBoost")
print(
    "Mean Recall:",
    round(
        xgb_cv_recall.mean(),
        4
    )
)

print(
    "Std:",
    round(
        xgb_cv_recall.std(),
        4
    )
)
# %%
# %% ============================================================
# CELL 29 - TEST SET PREDICTIONS
# ============================================================

svm_pred = best_svm.predict(
    X_test
)

rf_pred = best_rf.predict(
    X_test
)

xgb_pred = best_xgb.predict(
    X_test
)

svm_prob = best_svm.predict_proba(
    X_test
)[:, 1]

rf_prob = best_rf.predict_proba(
    X_test
)[:, 1]

xgb_prob = best_xgb.predict_proba(
    X_test
)[:, 1]

print("Predictions generated successfully.")
# %%
# %% ============================================================
# CELL 30 - MODEL EVALUATION FUNCTION
# ============================================================

def evaluate_model(
    name,
    y_true,
    y_pred,
    y_probability,
    cv_scores
):

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred
    ).ravel()

    sensitivity = recall_score(
        y_true,
        y_pred
    )

    specificity = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    result = {

        "Model": name,

        "Accuracy": accuracy_score(
            y_true,
            y_pred
        ),

        "Precision": precision_score(
            y_true,
            y_pred,
            zero_division=0
        ),

        "Recall / Sensitivity": sensitivity,

        "Specificity": specificity,

        "F1 Score": f1_score(
            y_true,
            y_pred
        ),

        "ROC-AUC": roc_auc_score(
            y_true,
            y_probability
        ),

        "PR-AUC": average_precision_score(
            y_true,
            y_probability
        ),

        "Mean CV Recall": np.mean(
            cv_scores
        ),

        "CV Recall Std": np.std(
            cv_scores
        )
    }

    return result
# %%
# %% ============================================================
# CELL 31 - EVALUATE ALL MODELS
# ============================================================

results = []

results.append(
    evaluate_model(
        "SVM",
        y_test,
        svm_pred,
        svm_prob,
        svm_cv_recall
    )
)

results.append(
    evaluate_model(
        "Random Forest",
        y_test,
        rf_pred,
        rf_prob,
        rf_cv_recall
    )
)

results.append(
    evaluate_model(
        "XGBoost",
        y_test,
        xgb_pred,
        xgb_prob,
        xgb_cv_recall
    )
)

results_df = pd.DataFrame(
    results
)

results_df = results_df.round(4)

print(results_df)
# %%
# %% ============================================================
# CELL 32 - SVM CLASSIFICATION REPORT
# ============================================================

print("SVM CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        svm_pred,
        target_names=[
            "No Diabetes",
            "Diabetes"
        ]
    )
)
# %%
# %% ============================================================
# CELL 33 - RANDOM FOREST CLASSIFICATION REPORT
# ============================================================

print("RANDOM FOREST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        rf_pred,
        target_names=[
            "No Diabetes",
            "Diabetes"
        ]
    )
)
# %%
# %% ============================================================
# CELL 34 - XGBOOST CLASSIFICATION REPORT
# ============================================================

print("XGBOOST CLASSIFICATION REPORT")
print("=" * 60)

print(
    classification_report(
        y_test,
        xgb_pred,
        target_names=[
            "No Diabetes",
            "Diabetes"
        ]
    )
)
# %%
# %% ============================================================
# CELL 35 - CONFUSION MATRIX FUNCTION
# ============================================================

def plot_confusion_matrix(
    matrix,
    title
):

    plt.figure(
        figsize=(5, 4)
    )

    plt.imshow(
        matrix
    )

    plt.title(title)

    plt.colorbar()

    tick_marks = np.arange(2)

    plt.xticks(
        tick_marks,
        ["No Diabetes", "Diabetes"]
    )

    plt.yticks(
        tick_marks,
        ["No Diabetes", "Diabetes"]
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")

    threshold = (
        matrix.max() / 2
    )

    for i in range(2):

        for j in range(2):

            plt.text(
                j,
                i,
                matrix[i, j],
                horizontalalignment="center",
                verticalalignment="center"
            )

    plt.tight_layout()
    plt.show()
# %%
# %% ============================================================
# CELL 36 - SVM CONFUSION MATRIX
# ============================================================

svm_cm = confusion_matrix(
    y_test,
    svm_pred
)

print(svm_cm)

plot_confusion_matrix(
    svm_cm,
    "SVM Confusion Matrix"
)
# %%
# %% ============================================================
# CELL 37 - RANDOM FOREST CONFUSION MATRIX
# ============================================================

rf_cm = confusion_matrix(
    y_test,
    rf_pred
)

print(rf_cm)

plot_confusion_matrix(
    rf_cm,
    "Random Forest Confusion Matrix"
)

# %%
# %% ============================================================
# CELL 38 - XGBOOST CONFUSION MATRIX
# ============================================================

xgb_cm = confusion_matrix(
    y_test,
    xgb_pred
)

print(xgb_cm)

plot_confusion_matrix(
    xgb_cm,
    "XGBoost Confusion Matrix"
)
# %%
# %% ============================================================
# CELL 39 - ROC CURVE COMPARISON
# ============================================================

svm_fpr, svm_tpr, _ = roc_curve(
    y_test,
    svm_prob
)

rf_fpr, rf_tpr, _ = roc_curve(
    y_test,
    rf_prob
)

xgb_fpr, xgb_tpr, _ = roc_curve(
    y_test,
    xgb_prob
)

plt.figure(
    figsize=(8, 6)
)

plt.plot(
    svm_fpr,
    svm_tpr,
    label=f"SVM AUC = {roc_auc_score(y_test, svm_prob):.3f}"
)

plt.plot(
    rf_fpr,
    rf_tpr,
    label=f"Random Forest AUC = {roc_auc_score(y_test, rf_prob):.3f}"
)

plt.plot(
    xgb_fpr,
    xgb_tpr,
    label=f"XGBoost AUC = {roc_auc_score(y_test, xgb_prob):.3f}"
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--"
)

plt.xlabel(
    "False Positive Rate"
)

plt.ylabel(
    "True Positive Rate / Sensitivity"
)

plt.title(
    "ROC Curves - Clinical Disease Prediction"
)

plt.legend()

plt.grid(
    alpha=0.3
)

plt.tight_layout()
plt.show()
# %%
# %% ============================================================
# CELL 40 - FINAL MODEL COMPARISON TABLE
# ============================================================

comparison_columns = [
    "Model",
    "Accuracy",
    "Precision",
    "Recall / Sensitivity",
    "Specificity",
    "F1 Score",
    "ROC-AUC",
    "PR-AUC",
    "Mean CV Recall",
    "CV Recall Std"
]

final_results = (
    results_df[
        comparison_columns
    ]
    .sort_values(
        by="Mean CV Recall",
        ascending=False
    )
)

print("\nFINAL MODEL COMPARISON")
print("=" * 100)

print(
    final_results.to_string(
        index=False
    )
)
# %%
# %% ============================================================
# CELL 41 - RECALL COMPARISON
# ============================================================

plt.figure(
    figsize=(8, 5)
)

plt.bar(
    results_df["Model"],
    results_df["Recall / Sensitivity"]
)

plt.ylim(
    0,
    1
)

plt.ylabel(
    "Recall / Sensitivity"
)

plt.title(
    "Test Recall Comparison"
)

for i, value in enumerate(
    results_df[
        "Recall / Sensitivity"
    ]
):

    plt.text(
        i,
        value + 0.02,
        f"{value:.3f}",
        ha="center"
    )

plt.tight_layout()
plt.show()
# %%
# %% ============================================================
# CELL 42 - CROSS-VALIDATION RECALL COMPARISON
# ============================================================

cv_data = [
    svm_cv_recall,
    rf_cv_recall,
    xgb_cv_recall
]

plt.figure(figsize=(8, 5))

plt.boxplot(
    cv_data,
    tick_labels=[
        "SVM",
        "Random Forest",
        "XGBoost"
    ]
)

plt.ylabel("Recall / Sensitivity")

plt.title(
    "5-Fold Cross-Validation Recall"
)

plt.ylim(0, 1)

plt.grid(
    axis="y",
    alpha=0.3
)

plt.tight_layout()
plt.show()
# %%
# %% ============================================================
# CELL 43 - SELECT BEST MODEL
# ============================================================

cv_summary = {

    "SVM": svm_cv_recall.mean(),

    "Random Forest": rf_cv_recall.mean(),

    "XGBoost": xgb_cv_recall.mean()
}

best_model_name = max(
    cv_summary,
    key=cv_summary.get
)

best_cv_recall = cv_summary[
    best_model_name
]

print("Selected Model:")
print(best_model_name)

print(
    "\nMean Cross-Validation Recall:",
    round(
        best_cv_recall,
        4
    )
)
# %%
# %% ============================================================
# CELL 44 - FINAL PROJECT SUMMARY
# ============================================================

best_row = (
    results_df[
        results_df["Model"]
        == best_model_name
    ]
    .iloc[0]
)

print("=" * 70)

print(
    "HIGH-IMBALANCE CLINICAL DISEASE PREDICTOR - FINAL SUMMARY"
)

print("=" * 70)

print(
    f"Selected model          : {best_model_name}"
)

print(
    f"Test accuracy           : {best_row['Accuracy']:.4f}"
)

print(
    f"Test precision          : {best_row['Precision']:.4f}"
)

print(
    f"Test sensitivity/recall : {best_row['Recall / Sensitivity']:.4f}"
)

print(
    f"Test specificity        : {best_row['Specificity']:.4f}"
)

print(
    f"Test F1 score           : {best_row['F1 Score']:.4f}"
)

print(
    f"Test ROC-AUC            : {best_row['ROC-AUC']:.4f}"
)

print(
    f"Mean 5-fold CV recall   : {best_row['Mean CV Recall']:.4f}"
)

print(
    f"CV recall std           : {best_row['CV Recall Std']:.4f}"
)

print("\nRecall CV arrays:")

print(
    "SVM:",
    np.round(
        svm_cv_recall,
        4
    )
)

print(
    "Random Forest:",
    np.round(
        rf_cv_recall,
        4
    )
)

print(
    "XGBoost:",
    np.round(
        xgb_cv_recall,
        4
    )
)

print("=" * 70)
# %%
# %% ============================================================
# CELL 45 - SAVE MODEL RESULTS
# ============================================================

final_results.to_csv(
    "task4_model_results.csv",
    index=False
)

print(
    "Results saved as: task4_model_results.csv"
)
# %%
# %% ============================================================
# CELL 46 - SAVE BEST MODEL
# ============================================================

import joblib

model_dictionary = {

    "SVM": best_svm,

    "Random Forest": best_rf,

    "XGBoost": best_xgb
}

final_model = model_dictionary[
    best_model_name
]

joblib.dump(
    final_model,
    "clinical_disease_predictor.pkl"
)

print(
    "Best model saved as:"
)

print(
    "clinical_disease_predictor.pkl"
)
# %%
# %% ============================================================
# CELL 47 - PREDICT A NEW PATIENT
# ============================================================

new_patient = pd.DataFrame(
    {
        "Pregnancies": [6],
        "Glucose": [148],
        "BloodPressure": [72],
        "SkinThickness": [35],
        "Insulin": [np.nan],
        "BMI": [33.6],
        "DiabetesPedigreeFunction": [0.627],
        "Age": [50]
    }
)

prediction = final_model.predict(
    new_patient
)[0]

probability = final_model.predict_proba(
    new_patient
)[0][1]

print(
    "Predicted Class:",
    prediction
)

print(
    "Estimated Positive-Class Probability:",
    round(
        probability,
        4
    )
)

if prediction == 1:

    print(
        "Model classification: Higher diabetes-risk pattern"
    )

else:

    print(
        "Model classification: Lower diabetes-risk pattern"
    )
# %%
