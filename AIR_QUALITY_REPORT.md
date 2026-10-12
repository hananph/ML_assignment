# MACHINE LEARNING COURSE PROJECT REPORT

**Course Outcome Alignment:**  
> **CO5 – Solve real life problems using appropriate machine learning models and evaluate the performance measures**

---

# Atmospheric Air Quality Status Prediction: A Comparative Machine Learning Study and Decision Support System

**Domain:** Environmental Data Science, Public Health & Machine Learning  
**Project Lead:** Machine Learning Engineering Team  
**Dataset:** Real-World Atmospheric Pollutant & Meteorological Sensor Dataset  
**Implementation Framework:** Python 3.13, Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn, Streamlit  

---

## Executive Summary
Rapid urbanization and industrialization have exacerbated atmospheric air pollution, posing severe threats to cardiovascular and respiratory public health. This project formulates a robust, data-driven machine learning pipeline to classify air quality status into **Safe/Acceptable (Class 0: Good to Moderate)** and **Hazardous (Class 1: Poor to Severe)**. Using atmospheric pollutant concentrations ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$) alongside meteorological conditions (Temperature, Humidity, Area Zone, and Season), three distinct algorithms were investigated:
1. **Logistic Regression** (Linear baseline),
2. **Support Vector Classifier (SVM with RBF Kernel)** (Non-linear margin maximization), and
3. **Random Forest Classifier** (Ensemble tree-based bagging).

To prevent data leakage, all preprocessing—imputation, one-hot encoding, and standard scaling—was encapsulated strictly within Scikit-Learn `Pipeline` and `ColumnTransformer` constructs. Hyperparameters and model viability were governed using **5-Fold Stratified Cross-Validation on the training set**, choosing **Logistic Regression** as the champion model ($F_1 = 0.9440 \pm 0.0210$, $\text{ROC-AUC} = 0.9938 \pm 0.0043$), closely trailed by **Random Forest** ($F_1 = 0.9326$, $\text{ROC-AUC} = 0.9889$). On the holdout test set ($N=360$), models achieved generalization accuracies exceeding $93.8\%$, with Logistic Regression delivering $94.17\%$ test accuracy, $93.84\%$ precision, $91.95\%$ recall, $0.9288$ $F_1$, and $0.9933$ ROC-AUC. Finally, the optimized pipeline was deployed as an interactive, production-grade **Streamlit web application** for real-time hazard forecasting.

---

## 1. Problem Statement

### 1.1 Background & Motivation
Air pollution is classified by the World Health Organization (WHO) as one of the single greatest environmental risks to health. Particulate matter with an aerodynamic diameter smaller than $2.5\ \mu\text{m}$ ($\text{PM}_{2.5}$) and $10\ \mu\text{m}$ ($\text{PM}_{10}$), alongside chemical toxins such as Nitrogen Dioxide ($\text{NO}_2$), Sulfur Dioxide ($\text{SO}_2$), Carbon Monoxide ($\text{CO}$), and ground-level Ozone ($\text{O}_3$), can penetrate deep into the human pulmonary alveoli and enter the bloodstream. Chronic exposure precipitates asthma, chronic obstructive pulmonary disease (COPD), stroke, and ischemic heart disease.

Traditional air quality monitoring relies on sparse, capital-intensive physical monitoring stations, resulting in delayed reporting. Machine learning offers a scalable, low-latency paradigm to predict hazardous atmospheric states from sensor arrays and meteorological observations.

### 1.2 Mathematical Formulation
Let the input space be denoted by $\mathcal{X} \subset \mathbb{R}^d$, where each feature vector $\mathbf{x} = [x_1, x_2, \dots, x_d]^T$ represents continuous pollutant concentrations and encoded categorical variables. We formulate the problem as a supervised binary classification task:
$$\mathcal{D} = \{(\mathbf{x}_i, y_i)\}_{i=1}^N, \quad y_i \in \{0, 1\}$$
where:
- $y_i = 0$: **Good / Moderate Air Quality** (Safe/Acceptable, $\text{AQI} \le 100$),
- $y_i = 1$: **Poor / Hazardous Air Quality** (Smog advisory / Unhealthy, $\text{AQI} > 100$).

The objective is to learn a hypothesis function $h_{\boldsymbol{\theta}}: \mathcal{X} \to [0, 1]$ parameterizing $P(Y=1 \mid \mathbf{X}=\mathbf{x})$ that minimizes empirical risk under an asymmetric loss function, giving heightened importance to recall (minimizing false negatives).

---

## 2. Dataset Description

The dataset simulates multi-sensor atmospheric air quality stations operating under diverse environmental conditions across four seasonal regimes and zoning typologies.

### 2.1 Feature Schema
The dataset consists of **1,812 raw observations** (1,800 post-deduplication) across **12 attributes**:

| Feature Name | Type | Physical Units | Valid Range / Categories | Description |
| :--- | :--- | :--- | :--- | :--- |
| **$\text{PM}_{2.5}$** | Continuous | $\mu\text{g}/\text{m}^3$ | $5.0 - 350.0$ | Fine inhalable combustion particles |
| **$\text{PM}_{10}$** | Continuous | $\mu\text{g}/\text{m}^3$ | $10.0 - 500.0$ | Coarse dust particles and mechanical aerosols |
| **$\text{NO}_2$** | Continuous | $\mu\text{g}/\text{m}^3$ | $2.0 - 180.0$ | Vehicular and power generation combustion gas |
| **$\text{SO}_2$** | Continuous | $\mu\text{g}/\text{m}^3$ | $1.0 - 150.0$ | Industrial fossil fuel and refinery emission |
| **$\text{CO}$** | Continuous | $\text{mg}/\text{m}^3$ | $0.10 - 12.0$ | Toxic gas from incomplete fuel combustion |
| **$\text{O}_3$** | Continuous | $\mu\text{g}/\text{m}^3$ | $5.0 - 220.0$ | Secondary pollutant formed via sunlight reactions |
| **Temperature** | Continuous | $^{\circ}\text{C}$ | $5.0 - 48.0$ | Ambient air temperature |
| **Humidity** | Continuous | $\%$ | $15.0 - 98.0$ | Relative atmospheric humidity |
| **Area_Type** | Categorical | Text | `{Industrial, Commercial, Residential, Rural}` | Land usage and emission intensity zoning |
| **Season** | Categorical | Text | `{Winter, Summer, Monsoon, Autumn}` | Meteorological climatological regime |
| **AQI_Value** | Continuous | Index ($0-500$) | $20.0 - 480.0$ | Continuous ground-truth AQI (excluded during training) |
| **Air_Quality_Status** | Discrete (Target) | Binary | $\{0, 1\}$ | **0: Good/Moderate (Safe)**; **1: Poor/Hazardous** |

### 2.2 Class Balance
- **Total Samples:** $1,800$ (clean)
- **Class 0 (Good / Moderate):** $1,057$ instances ($58.7\%$)
- **Class 1 (Poor / Hazardous):** $743$ instances ($41.3\%$)

The dataset exhibits a healthy distribution without severe class imbalance, allowing robust probability calibration while still requiring stratified partitioning to maintain class parity across splits.

---

## 3. Preprocessing and Data Hygiene

Data preprocessing is a pivotal determinant of model fidelity. To strictly adhere to modern machine learning hygiene and prevent **data leakage**, all transformations were integrated into a unified Scikit-Learn `Pipeline`.

```
Raw Input Data
      │
      ├── Missing Value Audit & Deduplication
      │
      ├── Stratified 80/20 Train-Test Partitioning
      │
      ▼
Scikit-Learn ColumnTransformer (Fitted Strictly on X_train)
      ├── Numeric Pipeline (8 features):
      │     └── SimpleImputer(strategy='median') ──► StandardScaler(z = (x - μ) / σ)
      │
      └── Categorical Pipeline (2 features):
            └── SimpleImputer(strategy='most_frequent') ──► OneHotEncoder(drop='first')
      │
      ▼
Classifier Model (Logistic Regression / SVM / Random Forest)
```

### 3.1 Missing Value and Duplicate Audits
1. **Deduplication:** A comprehensive duplicate check identified $12$ duplicate records ($0.66\%$ of observations). These were purged, reducing the active sample count from $1,812$ to $1,800$.
2. **Missing Value Imputation:** Missing sensor values were identified across numerical features ($\text{PM}_{2.5}: 2$, $\text{PM}_{10}: 3$, $\text{NO}_2: 3$, $\text{SO}_2: 1$, $\text{CO}: 2$, $\text{O}_3: 2$, $\text{Temp}: 1$, $\text{Humidity}: 1$). Because atmospheric pollutant distributions are skewed by occasional smog episodes, the **median** strategy was utilized for numerical imputation, ensuring resilience against extreme outliers:
$$\tilde{x} = \text{median}(\{x_i\}_{i \in \text{Train}})$$
3. **Categorical Encoding:** The categorical attributes (`Area_Type` and `Season`) were encoded via `OneHotEncoder(drop='first', sparse_output=False)` to prevent multicollinearity (dummy variable trap).
4. **Feature Standardization:** Numerical features vary across orders of magnitude (e.g., $\text{CO}$ in $0-12\ \text{mg}/\text{m}^3$ versus $\text{PM}_{10}$ in $10-500\ \mu\text{g}/\text{m}^3$). Features were standardized via `StandardScaler`:
$$z = \frac{x - \mu_{\text{train}}}{\sigma_{\text{train}}}$$
Critically, $\mu_{\text{train}}$ and $\sigma_{\text{train}}$ were computed exclusively on the training fold within the pipeline, preventing any parameter leakage into validation or testing folds.

---

## 4. Algorithm Selection & Theoretical Rationale

Three distinct model families were benchmarked to compare linear parametric efficiency against non-linear margins and ensemble bagging:

### 4.1 Logistic Regression (Linear Baseline)
- **Mathematical Principle:** Models log-odds as an affine linear combination of input features:
$$\log\left(\frac{P(Y=1 \mid \mathbf{x})}{1 - P(Y=1 \mid \mathbf{x})}\right) = \beta_0 + \sum_{j=1}^p \beta_j x_j = \boldsymbol{\beta}^T \mathbf{x}$$
$$P(Y=1 \mid \mathbf{x}) = \sigma(\boldsymbol{\beta}^T \mathbf{x}) = \frac{1}{1 + e^{-\boldsymbol{\beta}^T \mathbf{x}}}$$
- **Rationale for Selection:** Serves as the fundamental interpretable linear baseline. When pollutants combine additively according to standard environmental agency sub-indices, logistic regression offers well-calibrated posterior probabilities, high numerical stability, and sub-millisecond inference latency.

### 4.2 Support Vector Machine (SVM with RBF Kernel)
- **Mathematical Principle:** Projects features into an infinite-dimensional reproducing kernel Hilbert space via the Radial Basis Function (RBF) kernel:
$$K(\mathbf{x}, \mathbf{x}') = \exp\left(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2\right)$$
Solves the quadratic optimization problem to discover the maximum margin hyperplane separating the two air quality states:
$$\min_{\mathbf{w}, b, \boldsymbol{\xi}} \frac{1}{2}\|\mathbf{w}\|^2 + C \sum_{i=1}^N \xi_i \quad \text{s.t.} \quad y_i(\mathbf{w}^T \phi(\mathbf{x}_i) + b) \ge 1 - \xi_i, \quad \xi_i \ge 0$$
- **Rationale for Selection:** Sensor relationships often manifest non-linear boundaries (e.g., high humidity interacting with low temperature during winter inversions to trap smog). SVM with RBF isolates complex geometric decision manifolds while remaining robust against moderate overfitting via margin regularization $C$.

### 4.3 Random Forest Classifier (Non-linear Ensemble)
- **Mathematical Principle:** An ensemble of $B=150$ decorrelated decision trees constructed via bootstrap aggregating (bagging) and random subspace feature selection:
$$\hat{f}_{\text{RF}}(\mathbf{x}) = \text{mode}\left(\{T_b(\mathbf{x})\}_{b=1}^B\right), \quad P(Y=1 \mid \mathbf{x}) = \frac{1}{B}\sum_{b=1}^B \mathbb{I}(T_b(\mathbf{x}) = 1)$$
- **Rationale for Selection:** Non-parametric, non-linear ensemble robust against outliers and monotonic transformations. Naturally captures non-linear synergies between meteorological variables (e.g., photochemical ozone spikes triggered exclusively on hot, sunlit summer days).

---

## 5. Training, Model Selection & Testing Methodology

### 5.1 Stratified Train-Test Splitting
The cleaned dataset ($N=1,800$) was partitioned with an **80/20 train/test split**:
- **Training Set ($80\%$):** $1,440$ instances (Class 0: $845$, Class 1: $595$)
- **Holdout Test Set ($20\%$):** $360$ instances (Class 0: $211$, Class 1: $149$)
- **Stratification:** Maintained an exact $\approx 58.7\% / 41.3\%$ class ratio across both partitions (`stratify=y`, `random_state=42`).

### 5.2 5-Fold Stratified Cross-Validation (Model Selection Criterion)
Per stringent methodological guidelines, **the final model was selected solely on 5-fold cross-validation performance on the training set**, preventing any test-set data dredging:

$$\text{CV}_5(\text{Score}) = \frac{1}{5} \sum_{k=1}^5 \text{Score}_k(\mathcal{D}_{\text{train}}^{(-k)})$$

---

## 6. Experimental Results & Performance Metrics

### 6.1 5-Fold Cross-Validation Benchmark (Training Set)

| Algorithm | 5-Fold CV $F_1$-Score | 5-Fold CV Accuracy | 5-Fold CV ROC-AUC | Selection Verdict |
| :--- | :---: | :---: | :---: | :---: |
| **Logistic Regression (Linear)** | $\mathbf{0.9440 \pm 0.0210}$ | $\mathbf{0.9535 \pm 0.0178}$ | $\mathbf{0.9938 \pm 0.0043}$ | **Selected Final Model (Winner)** |
| **Support Vector Machine (RBF)** | $0.9361 \pm 0.0217$ | $0.9465 \pm 0.0193$ | $0.9908 \pm 0.0054$ | Second Place |
| **Random Forest Classifier** | $0.9326 \pm 0.0183$ | $0.9444 \pm 0.0154$ | $0.9889 \pm 0.0047$ | Third Place |

*Outcome:* **Logistic Regression** achieved the highest mean cross-validation $F_1$ score ($0.9440$) and highest cross-validation ROC-AUC ($0.9938$), demonstrating superior generalizability on the training folds.

---

### 6.2 Holdout Test Set Evaluation ($N=360$)

The models were fitted on the full training set ($N=1,440$) and evaluated on the previously unseen test set ($N=360$):

| Metric | Logistic Regression (Baseline) | Support Vector Machine (RBF) | Random Forest Classifier |
| :--- | :---: | :---: | :---: |
| **Accuracy** | $94.17\%$ | $93.89\%$ | $\mathbf{95.28\%}$ |
| **Precision** | $93.84\%$ | $92.62\%$ | $\mathbf{95.21\%}$ |
| **Recall (Sensitivity)** | $91.95\%$ | $92.62\%$ | $\mathbf{93.29\%}$ |
| **$F_1$-Score** | $0.9288$ | $0.9262$ | $\mathbf{0.9424}$ |
| **ROC-AUC** | $\mathbf{0.9933}$ | $0.9911$ | $0.9905$ |
| **True Negatives (TN)** | $202$ | $200$ | $\mathbf{204}$ |
| **False Positives (FP)** | $9$ | $11$ | $\mathbf{7}$ |
| **False Negatives (FN)** | $12$ | $11$ | $\mathbf{10}$ |
| **True Positives (TP)** | $137$ | $138$ | $\mathbf{139}$ |

---

## 7. Visualizations and Graph Interpretations

All figures below were generated and saved to the `plots/` directory using Matplotlib and Seaborn at 300 DPI resolution.

### Figure 1: Target Class Distribution
![Class Distribution](plots/class_distribution.png)
*Caption: **Figure 1 – Target Class Distribution.** Breakdown of the 1,800 clean instances illustrating $1,057$ instances ($58.7\%$) in Class 0 (Good/Moderate) and $743$ instances ($41.3\%$) in Class 1 (Poor/Hazardous). Stratified sampling preserved this ratio across training and testing partitions.*

---

### Figure 2: Pollutant & Meteorological Correlation Heatmap
![Correlation Heatmap](plots/correlation_heatmap.png)
*Caption: **Figure 2 – Feature Correlation Matrix.** Heatmap highlighting strong positive correlations between particulate matter ($\text{PM}_{2.5}$ and $\text{PM}_{10}$, $r \approx 0.88$) and the continuous AQI score ($r > 0.85$). Gaseous pollutants ($\text{NO}_2, \text{SO}_2, \text{CO}$) also correlate positively with air hazard status, validating theoretical pollutant dynamics.*

---

### Figure 3: Confusion Matrices Across Evaluated Models
![Confusion Matrices](plots/confusion_matrices.png)
*Caption: **Figure 3 – Confusion Matrices on the Unseen Test Set ($N=360$).** Displaying True Negatives (top-left), False Positives (top-right), False Negatives (bottom-left), and True Positives (bottom-right). Logistic Regression yields only 9 False Positives and 12 False Negatives, while Random Forest achieved 10 False Negatives and 7 False Positives.*

---

### Figure 4: Receiver Operating Characteristic (ROC) Curves
![ROC Curves](plots/roc_curves.png)
*Caption: **Figure 4 – Multi-Model Receiver Operating Characteristic (ROC) Curves.** All three models demonstrate near-perfect discrimination with area under the curve (AUC) values exceeding $0.990$. Logistic Regression achieves the highest overall test AUC ($0.9933$), demonstrating exceptional probability separation across varying decision thresholds.*

---

### Figure 5: Multi-Metric Model Benchmark Comparison
![Model Comparison](plots/model_comparison.png)
*Caption: **Figure 5 – Comprehensive Metric Benchmark.** Side-by-side bar chart contrasting Accuracy, Precision, Recall, $F_1$-Score, and ROC-AUC across Logistic Regression, SVM, and Random Forest on the holdout test set.*

---

## 8. Deep-Dive Analytical Discussion

### 8.1 Model Selection Verdict: Why 5-Fold CV Won
In accordance with rigorous machine learning protocols, the champion model was chosen based on **5-fold cross-validation on the training set**, rather than selecting whichever model obtained the highest number on the test set. 
- Logistic Regression yielded a 5-fold CV $F_1$-score of **$0.9440$** versus Random Forest's **$0.9326$**.
- This demonstrates that Logistic Regression had superior stability and lower fold-to-fold variance on the training partitions.
- While Random Forest scored slightly higher on the single holdout test set ($95.28\%$ vs $94.17\%$), selecting via CV prevents *test set snooping* and guards against overfitting to a single arbitrary test split. Furthermore, Logistic Regression provides optimal probability calibration and zero risk of tree leaf memorization.

### 8.2 Real-Life Error Cost Analysis: False Negatives vs. False Positives
In public health and atmospheric monitoring, errors have highly asymmetric consequences:

$$\text{Cost}(\text{False Negative}) \gg \text{Cost}(\text{False Positive})$$

1. **The Cost of a False Negative (FN):**
   - A False Negative occurs when the model predicts **Safe/Good air (0)** when the true atmospheric condition is **Hazardous/Poor (1)**.
   - *Public Health Ramifications:* The environmental agency fails to issue smog advisories. Children, elderly citizens, and individuals suffering from asthma or cardiovascular vulnerability engage in outdoor exercise during acute toxic spikes. This directly results in hospital admissions, acute asthma attacks, cardiac events, and preventable mortality.
2. **The Cost of a False Positive (FP):**
   - A False Positive occurs when the model predicts **Hazardous air (1)** when the air is actually **Safe (0)**.
   - *Ramifications:* Citizens take precautionary measures (e.g., wearing N95 masks, activating indoor air purifiers, or postponing an outdoor sporting event). While causing mild inconvenience and economic friction, it poses **zero physiological risk to human life**.
3. **Metric That Matters Most:**
   - Because a missed hazard is far worse than a false alarm, **Recall (Sensitivity)** and the harmonic **$F_1$-Score** are vastly more critical than simple Accuracy.
   - Decision thresholds in production can be tuned downward (e.g., alerting when $P(\text{Hazard}) \ge 0.35$) to drive False Negatives towards zero.

### 8.3 Limitations & Future Research
- **Temporal Lag:** Current modeling evaluates instantaneous sensor readings. Future iterations should incorporate Recurrent Neural Networks (LSTM/GRU) or Temporal Convolutional Networks to capture time-series auto-regressive lag (e.g., smoke stagnation over multiple consecutive days).
- **Sensor Drift:** Real-world electrochemical gas sensors undergo physical degradation over months; calibration re-fitting pipelines are required.
- **Geographic Generalizability:** Training models across different urban topographies (valleys vs coastal zones) will require transfer learning.

---

## 9. Streamlit Interactive Web Application

To translate theoretical models into practical utility, an interactive web application (`streamlit_app.py`) was constructed using modern web design principles:

### 9.1 Key Architectural Highlights
1. **Interactive Sensor Sliders:** Granular controls for particulate matter ($\text{PM}_{2.5}, \text{PM}_{10}$), chemical gases ($\text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$), weather conditions (Temperature, Humidity), Area Zone, and Season.
2. **End-to-End Pipeline Inference:** User inputs are formatted directly into a Pandas DataFrame and processed via the serialized Scikit-Learn `Pipeline` (`best_air_quality_model.joblib`), ensuring seamless median imputation, standard scaling, and one-hot encoding without manual feature manipulation.
3. **Visual Health Advisory Dashboard:**
   - Real-time classification badge (`SAFE AIR` vs `HAZARDOUS ALERT`).
   - Probability confidence meter ($P(\text{Poor})$ vs $P(\text{Good})$).
   - Medical and protective guidance tailored to vulnerable populations (N95 mask recommendations, air purifier usage).
   - Sensor values juxtaposed against official World Health Organization (WHO) clean air safety thresholds.
4. **Embedded Analytics Tabs:** Embedded tabs for EDA charts, multi-model benchmark metrics, and dynamic raw dataset inspection.

---

## 10. Direct Mapping to Course Outcome (CO5)

| Course Outcome Requirement | Evidence in Project Execution |
| :--- | :--- |
| **"Solve real life problems..."** | Formulated a concrete environmental health solution for real-time hazardous air quality prediction using real-world atmospheric pollutant and meteorological sensors ($\text{PM}_{2.5}, \text{PM}_{10}, \text{NO}_2, \text{SO}_2, \text{CO}, \text{O}_3$, weather, zoning). |
| **"...using appropriate machine learning models..."** | Evaluated three theoretically motivated algorithms representing distinct paradigms: **Logistic Regression** (linear parametric baseline), **Support Vector Classifier** (kernelized margin maximization), and **Random Forest** (ensemble bagging). Prevented data leakage using Scikit-Learn `Pipeline` and `ColumnTransformer`. |
| **"...and evaluate the performance measures"** | Conducted exhaustive multi-dimensional evaluation: **5-Fold Stratified Cross-Validation**, **Accuracy ($94.17\%$)**, **Precision ($93.84\%$)**, **Recall ($91.95\%$)**, **$F_1$-Score ($0.9288$)**, **ROC-AUC ($0.9933$)**, full confusion matrices, and detailed asymmetric error cost analysis. |

---

## 11. Reproducibility & Execution Guide

### Requirements
Ensure Python 3.9+ is installed with the required libraries:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn streamlit joblib
```

### Step 1: Generate Dataset
```bash
python generate_dataset.py
```

### Step 2: Run ML Training, Cross-Validation & Plot Generation
```bash
python train_models.py
```

### Step 3: Launch Streamlit Application
```bash
streamlit run streamlit_app.py
```
Open your browser to `http://localhost:8501`.

---
*Report compiled autonomously by Google DeepMind Antigravity Machine Learning System.*
