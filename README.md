
# Air Quality Prediction Machine Learning Project & Interactive Streamlit App

This repository provides an end-to-end Machine Learning pipeline and interactive web application for **Atmospheric Air Quality Status Prediction (Safe vs Hazardous)** based on environmental pollutant sensors and meteorological factors.

It strictly implements all five assignment steps, complete with cross-validation model selection, pipeline hygiene against data leakage, metric benchmarks, 5 publication-ready plots, a 4–6 page academic report, and a modern Streamlit application.

---

## 📂 Repository Structure

- [`generate_dataset.py`](file:///d:/ml%20assignment/generate_dataset.py): Realistic air quality dataset generator ($1,800$ clean rows with pollutants $\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$, temperature, humidity, missing values, duplicates, and text categories).
- [`air_quality_data.csv`](file:///d:/ml%20assignment/air_quality_data.csv): Generated atmospheric air quality dataset.
- [`train_models.py`](file:///d:/ml%20assignment/train_models.py): Complete end-to-end Machine Learning training script implementing:
  - Missing value imputation and duplicate removal
  - Text column encoding (`OneHotEncoder`)
  - Feature scaling with `StandardScaler` inside `Pipeline` to prevent data leakage
  - Stratified 80/20 train-test split (`stratify=y`, `random_state=42`)
  - 3 algorithms: **Logistic Regression** (linear baseline), **Support Vector Machine (SVM)**, **Random Forest Classifier**
  - **5-Fold Cross-Validation** on training set to choose the winning model
  - Test set metrics: Accuracy, Precision, Recall, F1, ROC-AUC, and Confusion Matrix
  - Generation and saving of all 5 plots
  - Model serialization to `best_air_quality_model.joblib` and `model.joblib`
- [`AIR_QUALITY_REPORT.md`](file:///d:/ml%20assignment/AIR_QUALITY_REPORT.md): Comprehensive academic research report (4–6 pages) with problem statement, dataset description, preprocessing, algorithm selection, metrics table, captions, error analysis (False Negatives vs False Positives), limitations, and mapping to **CO5**.
- [`streamlit_app.py`](file:///d:/ml%20assignment/streamlit_app.py): Interactive Streamlit application featuring live sliders, instant predictions, confidence meters, health advisories, WHO guidelines comparison, and embedded EDA dashboards.
- [`plots/`](file:///d:/ml%20assignment/plots): Publication-ready figures saved via `plt.savefig()`:
  - `class_distribution.png`: Target class distribution.
  - `correlation_heatmap.png`: Feature correlation matrix.
  - `confusion_matrices.png`: Confusion matrices comparing all 3 models.
  - `roc_curves.png`: Multi-model ROC curves with AUC scores.
  - `model_comparison.png`: Metric comparison bar chart.
- [`model_evaluation_results.json`](file:///d:/ml%20assignment/model_evaluation_results.json): Benchmark metrics recorded across models and cross-validation folds.

---

## 🚀 How to Run the Project

### 1. Install Dependencies
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### 2. Generate Dataset
```bash
python generate_dataset.py
```

### 3. Run Training, Cross-Validation & Plot Generation
```bash
python train_models.py
```

### 4. Launch the Streamlit Web Application
```bash
streamlit run streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.
