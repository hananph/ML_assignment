"""
Air Quality Classification & Model Comparison Pipeline
Following Machine Learning Assignment Specification:
Step 3:
1. Load data and print shape, column names, class counts
2. Preprocess: missing values, duplicates, encode text, scale with StandardScaler inside Pipeline
3. Split 80/20 with train_test_split(..., stratify=y, random_state=42)
4. Compare 3 algorithms:
   - Logistic Regression (Linear baseline)
   - Support Vector Machine (SVM / SVC with RBF)
   - Random Forest Classifier (Non-linear Ensemble)
5. 5-Fold Cross Validation on training set to choose the winning model
6. Evaluate on test set: Accuracy, Precision, Recall, F1, ROC-AUC, Confusion Matrix
7. Generate and save all 5 plots:
   - class_distribution.png
   - correlation_heatmap.png
   - confusion_matrices.png
   - roc_curves.png
   - model_comparison.png
"""

import json
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# Output directory for plots
PLOTS_DIR = "plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set clean aesthetic style for matplotlib / seaborn
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "figure.titlesize": 14,
        "figure.titleweight": "bold",
    }
)


def main():
    print("=" * 80)
    print("STEP 3: AIR QUALITY MACHINE LEARNING PIPELINE")
    print("=" * 80)

    # 1. LOAD DATA
    data_path = "air_quality_data.csv"
    if not os.path.exists(data_path):
        raise FileNotFoundError(
            f"{data_path} not found! Please run generate_dataset.py first."
        )

    df = pd.read_csv(data_path)
    print("\n[1] DATASET OVERVIEW:")
    print(f"Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print("\nClass distribution for 'Air_Quality_Status':")
    class_counts = df["Air_Quality_Status"].value_counts()
    print(class_counts)
    print(
        f"Class 0 (Good/Moderate): {class_counts.get(0, 0)} ({class_counts.get(0, 0) / len(df) * 100:.1f}%)"
    )
    print(
        f"Class 1 (Poor/Hazardous): {class_counts.get(1, 0)} ({class_counts.get(1, 0) / len(df) * 100:.1f}%)"
    )

    # 2. PREPROCESSING & DATA INTEGRITY
    print("\n[2] PREPROCESSING CHECKS:")
    # Missing values check
    missing = df.isnull().sum()
    print("Missing values per column:")
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values found.")

    # Duplicates check
    duplicates_count = df.duplicated().sum()
    print(f"Duplicate rows detected: {duplicates_count}")
    if duplicates_count > 0:
        df = df.drop_duplicates().reset_index(drop=True)
        print(f"Duplicates removed. Cleaned dataset shape: {df.shape}")

    # Feature definitions
    # Note: AQI_Value is the continuous index from which status was calculated,
    # so we exclude it from features to prevent target leakage!
    num_features = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "Temperature",
        "Humidity",
    ]
    cat_features = ["Area_Type", "Season"]
    target_col = "Air_Quality_Status"

    X = df[num_features + cat_features].copy()
    y = df[target_col].copy()

    # Plot 1: Class Distribution
    plt.figure(figsize=(7, 5))
    ax = sns.countplot(x=y, hue=y, palette=["#2ecc71", "#e74c3c"], legend=False)
    plt.title("Air Quality Target Class Distribution")
    plt.xlabel("Air Quality Status (0 = Good/Moderate, 1 = Poor/Hazardous)")
    plt.ylabel("Sample Count")
    plt.xticks(
        [0, 1],
        ["0: Good / Moderate\n(Acceptable Air)", "1: Poor / Hazardous\n(Smog Warning)"],
    )
    for p in ax.patches:
        height = int(p.get_height())
        ax.annotate(
            f"{height}\n({height / len(y) * 100:.1f}%)",
            (p.get_x() + p.get_width() / 2.0, height / 2),
            ha="center",
            va="center",
            color="white",
            fontweight="bold",
            fontsize=11,
        )
    plt.tight_layout()
    plot1_path = os.path.join(PLOTS_DIR, "class_distribution.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved: {plot1_path}")

    # Plot 2: Correlation Heatmap
    plt.figure(figsize=(9, 7))
    corr_cols = num_features + ["AQI_Value", "Air_Quality_Status"]
    corr_matrix = df[corr_cols].corr()
    np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar_kws={"label": "Correlation Coefficient"},
        linewidths=0.5,
        linecolor="white",
    )
    plt.title("Pollutant & Meteorological Feature Correlation Heatmap")
    plt.tight_layout()
    plot2_path = os.path.join(PLOTS_DIR, "correlation_heatmap.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved: {plot2_path}")

    # 3. SPLIT 80/20 WITH STRATIFY
    print("\n[3] TRAIN / TEST SPLIT:")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=42
    )
    print(
        f"Training set: {X_train.shape[0]} samples (Class 0: {(y_train == 0).sum()}, Class 1: {(y_train == 1).sum()})"
    )
    print(
        f"Testing set:  {X_test.shape[0]} samples (Class 0: {(y_test == 0).sum()}, Class 1: {(y_test == 1).sum()})"
    )

    # 4. PREPROCESSING PIPELINE (Inside Pipeline to avoid data leakage)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(
                    drop="first", handle_unknown="ignore", sparse_output=False
                ),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )

    # 5. ALGORITHM SELECTION & RATIONALE
    print("\n[4] ALGORITHM SELECTION & RATIONALE:")
    print(
        "1. Logistic Regression: Linear parametric baseline. Offers high interpretability, calibrated probabilities,"
    )
    print("   and fast inference. Assumes linear relationship in the log-odds space.")
    print(
        "2. Support Vector Machine (RBF Kernel): Non-linear boundary classifier that maximizes the separation margin."
    )
    print(
        "   Captures complex boundaries in transformed feature space via kernel trick."
    )
    print(
        "3. Random Forest Classifier: Non-linear ensemble of decision trees. "
        "Captures non-linear pollutant interactions,"
    )
    print(
        "   naturally resistant to scaling quirks, robust against variance, provides feature importances."
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Support Vector Machine": SVC(kernel="rbf", probability=True, random_state=42),
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=10, random_state=42
        ),
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # 6. 5-FOLD CROSS-VALIDATION ON TRAINING SET
    print("\n[5] 5-FOLD CROSS-VALIDATION (Training Set Only):")
    cv_results = {}
    pipelines = {}

    for name, model in models.items():
        pipe = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", model)])
        pipelines[name] = pipe

        # Cross-validation scores (F1 and ROC-AUC)
        cv_f1 = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="f1")
        cv_acc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="accuracy")
        cv_auc = cross_val_score(pipe, X_train, y_train, cv=cv, scoring="roc_auc")

        cv_results[name] = {
            "cv_f1_mean": float(np.mean(cv_f1)),
            "cv_f1_std": float(np.std(cv_f1)),
            "cv_acc_mean": float(np.mean(cv_acc)),
            "cv_acc_std": float(np.std(cv_acc)),
            "cv_auc_mean": float(np.mean(cv_auc)),
            "cv_auc_std": float(np.std(cv_auc)),
        }
        print(f"-> {name}:")
        print(
            f"   CV F1-Score: {cv_results[name]['cv_f1_mean']:.4f} (+/- {cv_results[name]['cv_f1_std']:.4f})"
        )
        print(
            f"   CV Accuracy: {cv_results[name]['cv_acc_mean']:.4f} (+/- {cv_results[name]['cv_acc_std']:.4f})"
        )
        print(
            f"   CV ROC-AUC:  {cv_results[name]['cv_auc_mean']:.4f} (+/- {cv_results[name]['cv_auc_std']:.4f})"
        )

    # Select final model based strictly on 5-fold CV F1 score (not test score)
    best_model_name = max(cv_results, key=lambda k: cv_results[k]["cv_f1_mean"])
    print(f"\n>>> FINAL MODEL SELECTED VIA 5-FOLD CROSS-VALIDATION: {best_model_name}")

    # 7. FIT AND TEST SET EVALUATION
    print("\n[6] TEST SET EVALUATION (Out-of-sample Generalization):")
    test_metrics = {}
    test_predictions = {}
    test_probabilities = {}
    cms = {}

    for name, pipe in pipelines.items():
        # Fit on full training set
        pipe.fit(X_train, y_train)

        # Predict on test set
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred)

        test_metrics[name] = {
            "Accuracy": float(acc),
            "Precision": float(prec),
            "Recall": float(rec),
            "F1_Score": float(f1),
            "ROC_AUC": float(auc),
            "TN": int(cm[0, 0]),
            "FP": int(cm[0, 1]),
            "FN": int(cm[1, 0]),
            "TP": int(cm[1, 1]),
        }
        test_predictions[name] = y_pred
        test_probabilities[name] = y_prob
        cms[name] = cm

        print(f"\n--- {name} ---")
        print(f"Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
        print(f"Precision: {prec:.4f}")
        print(f"Recall:    {rec:.4f}")
        print(f"F1-Score:  {f1:.4f}")
        print(f"ROC-AUC:   {auc:.4f}")
        print("Confusion Matrix:")
        print(cm)
        print(f"True Negatives: {cm[0, 0]} | False Positives: {cm[0, 1]}")
        print(f"False Negatives: {cm[1, 0]} | True Positives: {cm[1, 1]}")

    # Plot 3: Confusion Matrices for All 3 Models
    _fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for idx, (name, cm) in enumerate(cms.items()):
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            cbar=False,
            ax=axes[idx],
            xticklabels=["0: Good", "1: Poor"],
            yticklabels=["0: Good", "1: Poor"],
            annot_kws={"size": 13, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{name}\nAcc: {test_metrics[name]['Accuracy']:.3f} | F1: {test_metrics[name]['F1_Score']:.3f}"
        )
        axes[idx].set_xlabel("Predicted Label")
        axes[idx].set_ylabel("True Label")
    plt.suptitle(
        "Confusion Matrices Comparison (Test Set N=360)", fontsize=15, fontweight="bold"
    )
    plt.tight_layout()
    plot3_path = os.path.join(PLOTS_DIR, "confusion_matrices.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved: {plot3_path}")

    # Plot 4: ROC Curves for All Models
    plt.figure(figsize=(8, 6))
    colors = ["#2980b9", "#8e44ad", "#27ae60"]
    for idx, (name, y_prob) in enumerate(test_probabilities.items()):
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        auc = test_metrics[name]["ROC_AUC"]
        plt.plot(fpr, tpr, color=colors[idx], lw=2.5, label=f"{name} (AUC = {auc:.3f})")
    plt.plot(
        [0, 1],
        [0, 1],
        color="grey",
        lw=1.5,
        linestyle="--",
        label="Random Chance (AUC = 0.50)",
    )
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate (1 - Specificity)")
    plt.ylabel("True Positive Rate (Recall / Sensitivity)")
    plt.title("Receiver Operating Characteristic (ROC) Curves")
    plt.legend(loc="lower right", frameon=True, facecolor="white", framealpha=0.9)
    plt.tight_layout()
    plot4_path = os.path.join(PLOTS_DIR, "roc_curves.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved: {plot4_path}")

    # Plot 5: Model Comparison Bar Chart
    metrics_to_plot = ["Accuracy", "Precision", "Recall", "F1_Score", "ROC_AUC"]
    plot_data = []
    for model_name, m in test_metrics.items():
        for metric_name in metrics_to_plot:
            plot_data.append(
                {"Model": model_name, "Metric": metric_name, "Score": m[metric_name]}
            )
    plot_df = pd.DataFrame(plot_data)

    plt.figure(figsize=(10, 6))
    ax = sns.barplot(
        data=plot_df,
        x="Metric",
        y="Score",
        hue="Model",
        palette=["#3498db", "#9b59b6", "#2ecc71"],
    )
    plt.title("Comprehensive Model Comparison on Test Set")
    plt.ylim(0.70, 1.02)
    plt.ylabel("Score (0.0 to 1.0)")
    plt.xlabel("Evaluation Metric")
    plt.legend(title="Algorithm", loc="lower right")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(
                f"{h:.3f}",
                (p.get_x() + p.get_width() / 2.0, h),
                ha="center",
                va="bottom",
                fontsize=8,
                rotation=0,
            )
    plt.tight_layout()
    plot5_path = os.path.join(PLOTS_DIR, "model_comparison.png")
    plt.savefig(plot5_path, dpi=300)
    plt.close()
    print(f"Saved: {plot5_path}")

    # 8. SAVE TRAINED MODEL & METADATA
    best_pipeline = pipelines[best_model_name]
    model_filename = "best_air_quality_model.joblib"
    joblib.dump(best_pipeline, model_filename)
    # Also dump as model.joblib for generic accessibility
    joblib.dump(best_pipeline, "model.joblib")
    print(f"\n[7] WINNING MODEL SAVED to '{model_filename}' and 'model.joblib'")

    # Save summary evaluation json for the report & app
    summary_data = {
        "best_model": best_model_name,
        "cv_results": cv_results,
        "test_metrics": test_metrics,
        "num_features": num_features,
        "cat_features": cat_features,
        "dataset_rows": len(df),
        "test_rows": len(y_test),
        "train_rows": len(y_train),
    }
    with open("model_evaluation_results.json", "w") as f:
        json.dump(summary_data, f, indent=4)
    print("Saved evaluation results to 'model_evaluation_results.json'")

    print("\n" + "=" * 80)
    print("TRAINING AND EVALUATION COMPLETE SUCCESSFULLY!")
    print("=" * 80)


if __name__ == "__main__":
    main()
