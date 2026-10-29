# Task 3: Machine Learning Classifier Comparison and Personal Calibration Experiments

**Author:** Pirushalini  
**Project:** Sign Language Detector (BSL Two-Handed Fingerspelling)  
**Repository Branch:** `feature/ml-classifier-comparison`  
**Date:** October 2026  

---

## 1. Executive Summary

This report presents the complete research, implementation, evaluation, and application integration for **Task 3: Machine Learning Classifier Training, Evaluation, Comparison, and Personal Calibration Experiments**.

The existing desktop application recognized 26 British Sign Language (BSL) fingerspelling letters using an instance-based **Template Matcher** over 132-dimensional normalized hand landmark vectors. The supervisor requested an investigation into alternative machine learning classifiers to determine whether statistical models can improve recognition accuracy, resolve ambiguities between visually similar signs, reduce inference latency, and function effectively under personal calibration constraints.

Four supervised machine learning classifiers were developed, cross-validated, trained, and benchmarked against the Template Matcher baseline:
1. **K-Nearest Neighbors (KNN)**
2. **Support Vector Machine (SVM with RBF kernel and Platt probability calibration)**
3. **Random Forest (Ensemble of 200 decision trees)**
4. **Multi-Layer Perceptron (MLP Neural Network)**

### Primary Experimental Findings

| Classifier | Test Accuracy | Macro F1 | Mean Latency (ms) | Median Latency (ms) | p95 Latency (ms) | ECE (Calibration Error) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Template Matcher (Baseline)** | 99.67% | 0.9967 | 0.175 ms | 0.172 ms | 0.186 ms | N/A (Distances) |
| **KNN (k=5, Distance-Weighted)** | 99.18% | 0.9918 | 1.448 ms | 1.429 ms | 1.968 ms | 0.00930 |
| **SVM (RBF, C=10, Calibrated)** | **99.95%** | **0.9995** | 6.852 ms | 6.817 ms | 7.123 ms | 0.07735 |
| **Random Forest (200 trees)** | 99.62% | 0.9962 | 13.239 ms | 13.553 ms | 13.946 ms | 0.06052 |
| **MLP (256-128 Hidden Units)** | 99.35% | 0.9935 | **0.158 ms** | **0.156 ms** | **0.174 ms** | 0.02469 |

- **Highest Accuracy:** **SVM (RBF)** achieved **99.95% accuracy** on the held-out test split, misclassifying only 1 sample out of 1,840 test frames ($U \rightarrow T$).
- **Fastest Inference:** **MLP** achieved **0.158 ms per frame** (~6,300 frames per second), outperforming Template Matching (0.175 ms) and running 43x faster than SVM and 84x faster than Random Forest.
- **Cross-Signer Generalization Reality:** Zero-shot evaluation of public dataset models on personal calibration samples showed a steep performance drop (**41.76%** for Template Matcher, **36.76%** for SVM). Hand size, camera geometry, and signer positioning differences prove that personal calibration is **essential**.
- **Personal Calibration Efficacy:** A classifier trained **exclusively** on 40 personal calibration samples per letter achieved **100.0% accuracy** (Template Matcher and Random Forest) and **98.75% accuracy** (SVM). Just 10 personal samples per letter (~1.2 seconds of recording) yields **99.8%** personal accuracy.
- **Live Integration:** A modular, duck-type compatible `MLBackend` was created, enabling live runtime switching between Template Matching and Machine Learning via the Settings menu without degrading responsiveness.

---

## 2. Existing Recognition System

The application processes live camera frames through the following pipeline:
```
Camera Frame (1280x720) 
  ──> MediaPipe Hands (Static Image Mode=False, Max 2 Hands) 
  ──> Landmark Extraction (21 3D points per hand) 
  ──> build_features() (132-dimensional invariant vector) 
  ──> Classifier Backend (Template Matcher or MLBackend) 
  ──> Smoother (7-frame majority voting) 
  ──> HoldToAdd (1.0 s thresholded stability timer) 
  ──> Recognized Character Appended to Text Display
```

### Feature Vector Layout (132 Dimensions)

The 132-dimensional feature vector defined in `utils.py` provides translation, scale, and camera-distance invariance while preserving two-hand spatial relationships:

1. **Hand 1 Features (`[0:63]`, 63 values):**
   - 21 3D landmarks relative to Hand 1's wrist: $\mathbf{p}_i - \mathbf{p}_{\text{wrist}}$.
   - Multiplied by camera aspect ratio: $x' = x \cdot \text{aspect}$, $z' = z \cdot \text{aspect}$.
   - Normalized by maximum 2D radial span: $\text{scale}_1 = \max_i \|\mathbf{p}_i^{(x,y)} - \mathbf{p}_{\text{wrist}}^{(x,y)}\|_2$.
   - Zeros if Hand 1 is not detected.
2. **Hand 2 Features (`[63:126]`, 63 values):**
   - Identical wrist-relative normalization for Hand 2.
   - Hands are ordered strictly left-to-right by wrist x-coordinate so hand identity is invariant to MediaPipe tracking jitter.
3. **Inter-Hand Relationship (`[126:130]`, 4 values):**
   - Wrist-to-wrist displacement: $\Delta \mathbf{w} = (\mathbf{w}_2 - \mathbf{w}_1) / \overline{\text{scale}}$.
   - Euclidean wrist distance: $\|\Delta \mathbf{w}\|_2$.
   - Critical for letters where the dominant index finger touches a specific location on the non-dominant palm (e.g. vowels A, E, I, O, U, and consonants D, M, N).
4. **Presence Flags (`[130:132]`, 2 values):**
   - Binary flags $[h_1, h_2] \in \{0.0, 1.0\}^2$ indicating whether Hand 1 and Hand 2 are present.

### Baseline Template Matcher

The baseline `TemplateMatcher` in `matcher.py` stores unweighted landmark vectors. Inference proceeds as:
- Features are scaled by feature-type weights ($\sqrt{w_{\text{inter}}} = \sqrt{2.0}$, $\sqrt{w_{\text{flags}}} = \sqrt{3.0}$).
- Distances to personal templates are multiplied by `user_weight = 0.85` (prioritizing personal recordings).
- Letter scores equal the average Euclidean distance to the $k=3$ nearest templates for each class.
- Probabilities are computed via Boltzmann softmax: $p_c = \frac{\exp(-(s_c - s_{\min})/\tau)}{\sum_j \exp(-(s_j - s_{\min})/\tau)}$ with temperature $\tau = 0.12$.
- Matches with distance $> \text{reject\_distance}$ (default 1.5) return `None`.

---

## 3. Machine Learning Classifiers

Four classifiers from `scikit-learn` were selected to explore distinct algorithmic paradigms:

### A. K-Nearest Neighbors (KNN)
- **Algorithm:** Non-parametric instance-based classification. Finds the $k$ training points closest in Euclidean distance to the query vector.
- **Why Selected:** Natural statistical extension of template matching. Evaluates whether distance-weighting over larger neighborhoods ($k=5$) improves stability.
- **Training & Preprocessing:** Wrapped in a `Pipeline([('scaler', StandardScaler()), ('clf', KNeighborsClassifier())])`. The scaler standardizes each feature to zero mean and unit variance based strictly on training data.
- **Best Configuration:** $k=5$, distance weighting ($w_i = 1/d_i$), Euclidean metric.
- **Strengths:** Robust to multi-modal class clusters; no assumptions regarding linear separability.
- **Weaknesses:** Memory grows linearly with dataset size; inference requires distance computations across all stored vectors (1.45 ms).

### B. Support Vector Machine (SVM)
- **Algorithm:** Maximum-margin hyperplane classifier mapped into infinite-dimensional Hilbert space via the Radial Basis Function (RBF) kernel:
  $$K(\mathbf{x}, \mathbf{x}') = \exp(-\gamma \|\mathbf{x} - \mathbf{x}'\|^2)$$
- **Why Selected:** Industry gold standard for high-dimensional, small-to-medium sample landmark classification. Maximizes the geometric margin between confusing handshapes.
- **Probability Calibration:** Raw SVM outputs are uncalibrated decision distances. To obtain trustworthy confidence probabilities, Platt scaling (`CalibratedClassifierCV(method='sigmoid', cv=3)`) was fitted on training folds.
- **Best Configuration:** Kernel: `rbf`, $C=10.0$, $\gamma=\text{'scale'}$, with 3-fold Platt calibration.
- **Strengths:** Exceptional generalization boundary; achieved the highest overall accuracy (99.95%).
- **Weaknesses:** Quadratic training complexity; inference requires computing kernel evaluations against support vectors (6.85 ms).

### C. Random Forest
- **Algorithm:** Ensemble of 200 de-correlated CART decision trees trained with bagging and random feature subspace sampling ($\sqrt{132} \approx 11$ features per split).
- **Why Selected:** Robust to non-linear feature interactions, invariant to monotonic feature scaling, and provides native probabilistic voting across trees.
- **Best Configuration:** `n_estimators=200`, `max_depth=None`, `min_samples_leaf=1`, `class_weight='balanced'`.
- **Strengths:** Excellent resistance to outliers; handles discrete presence flags seamlessly.
- **Weaknesses:** Traversing 200 deep decision trees sample-by-sample induces high latency (13.24 ms per frame).

### D. Multi-Layer Perceptron (MLP)
- **Algorithm:** Feedforward artificial neural network with two fully connected hidden layers:
  $$\mathbf{x} \in \mathbb{R}^{132} \xrightarrow{\text{Dense}} \mathbf{h}_1 \in \mathbb{R}^{256} \xrightarrow{\text{ReLU}} \mathbf{h}_2 \in \mathbb{R}^{128} \xrightarrow{\text{ReLU}} \mathbf{y} \in \mathbb{R}^{23} \xrightarrow{\text{Softmax}}$$
- **Why Selected:** Parametric neural representation. Once weights are learned, inference requires only two matrix multiplications, making it exceptionally fast.
- **Regularization & Early Stopping:** Adam optimizer ($\text{lr}=10^{-3}$), $L_2$ weight decay ($\alpha=10^{-3}$), early stopping monitored on a 10% validation hold-out to prevent overfitting small datasets.
- **Best Configuration:** Hidden layers: `(256, 128)`, Activation: `ReLU`, $\alpha=0.001$, `early_stopping=True`.
- **Strengths:** Ultra-fast inference (0.158 ms); compact model size (850 KB); calibrated softmax probabilities.
- **Weaknesses:** Requires iterative optimization; sensitive to initial weight seed on tiny sample sets.

---

## 4. Dataset Preparation & Leakage Prevention

### Available Datasets
1. **External Public Dataset (`data/templates_external.npz`):**
   - 9,200 total samples across 23 letters (400 samples per letter; letters H, J, Y missing).
   - Source: Kaggle / Brunel University final-year project by datMaul / erentatepe.
   - Captured from a single signer under fixed camera framing.
2. **Personal User Calibration Dataset (`data/templates_user.npz`):**
   - 1,200 total samples across 20 letters (60 samples per letter).
   - Captured via `calibrate.py` across 7 recorded sessions (`day1`, `session_1951`, `session_2027`, `session_2041`, `session_2047`, `session_2048`, `session_2126`).

### Leakage Prevention Protocol
1. **Pipeline Isolation:** All scalers (`StandardScaler`) are encapsulated within `sklearn.pipeline.Pipeline` objects and fitted exclusively on training splits. Test vectors are strictly transformed using training statistics.
2. **Held-Out Stratified Partition (External Data):**
   - 80% Training ($N=7,360$, 320 samples/class).
   - 20% Held-Out Testing ($N=1,840$, 80 samples/class).
   - A fixed random seed (`SEED=42`) guarantees exact reproducibility.
3. **Personal Calibration Block Partition:**
   - Samples 0..39 (40 frames, 66.7%) assigned to calibration training.
   - Samples 40..59 (20 frames, 33.3%) assigned to held-out evaluation.
   - Temporal block splitting prevents interleaving adjacent camera frames.

---

## 5. Experimental Results: External Dataset Benchmark

All models were evaluated on the identical held-out test split of 1,840 samples across 23 classes.

### Performance Summary Table

| Metric | Template Matcher | KNN | SVM (RBF) | Random Forest | MLP |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Accuracy** | 99.67% | 99.18% | **99.95%** | 99.62% | 99.35% |
| **Macro F1** | 0.9967 | 0.9918 | **0.9995** | 0.9962 | 0.9935 |
| **Macro Precision** | 0.9968 | 0.9922 | **0.9995** | 0.9963 | 0.9936 |
| **Macro Recall** | 0.9967 | 0.9918 | **0.9995** | 0.9962 | 0.9935 |
| **Mean Inference Time** | 0.175 ms | 1.448 ms | 6.852 ms | 13.239 ms | **0.158 ms** |
| **Median Inference Time** | 0.172 ms | 1.429 ms | 6.817 ms | 13.553 ms | **0.156 ms** |
| **95th %ile Latency** | 0.186 ms | 1.968 ms | 7.123 ms | 13.946 ms | **0.174 ms** |
| **Expected Calib. Error (ECE)** | N/A | **0.00930** | 0.07735 | 0.06052 | 0.02469 |
| **Mean Top-2 Confidence Margin**| N/A | 0.9714 | 0.8543 | 0.9037 | 0.9450 |

### Top Confused Letter Pairs
- **SVM:** $U \rightarrow T$ (1 error total across 1,840 test frames).
- **Template Matcher:** $V \rightarrow N$ (1), $V \rightarrow L$ (1), $U \rightarrow T$ (1), $U \rightarrow O$ (1), $U \rightarrow E$ (1).
- **KNN:** $U \rightarrow O$ (4), $O \rightarrow U$ (3), $I \rightarrow O$ (2), $V \rightarrow L$ (1).
- **Random Forest:** $U \rightarrow O$ (3), $V \rightarrow L$ (1), $T \rightarrow U$ (1), $L \rightarrow M$ (1).
- **MLP:** $V \rightarrow N$ (3), $E \rightarrow I$ (2), $V \rightarrow U$ (1), $U \rightarrow T$ (1).

### Observations
1. **SVM demonstrates superior classification separation:** The non-linear RBF kernel mapped subtle wrist-angle differences between touching-hand vowels ($U, O, I$) that caused minor confusion in distance-based nearest neighbors.
2. **Inference Latency Profile:**
   - MLP is the fastest (0.158 ms), slightly edging out Template Matching (0.175 ms).
   - SVM requires 6.85 ms per sample, which is within the 33 ms real-time budget (30 FPS), but noticeably higher than MLP.
   - Random Forest is the slowest (13.24 ms) due to traversing 200 trees.
3. **Probability Calibration:**
   - KNN with distance weighting achieved the lowest Expected Calibration Error ($\text{ECE} = 0.0093$).
   - Platt-calibrated SVM produced confident margins (mean margin 0.854), providing clean discrimination thresholds for Task 4.

---

## 6. Personal Calibration Experiments

The supervisor designated the personal calibration experiment as a mandatory investigative component.

### Session Diagnostic & Limitation Analysis
Inspection of `data/templates_user.npz` revealed an important structural characteristic:
- **7 recording sessions:** `day1` (B, K, S), `session_1951` (J, Y), `session_2027` (E, F, L, M, N, O, Q, T, U, W), `session_2041` (A), `session_2047` (R), `session_2048` (V), `session_2126` (H, I).
- **Pairwise class overlap between sessions:** **0%**. Every letter was calibrated in only one session.
- **Scientific Conclusion:** A true Leave-One-Session-Out (LOSO) cross-validation across all letters is impossible on this archive because holding out any session removes all training examples for its constituent letters.
- **Methodology Applied:** To evaluate personal recognition without fabricating data or introducing leakage, each letter's 60 frames were split into calibration (samples 0..39) and held-out test (samples 40..59). All 4 experiments were evaluated on this exact test set.

### Experiment A: General Dataset Only (Zero-Shot Cross-Signer Transfer)
- **Training:** Public dataset (6,800 samples, 17 shared letters).
- **Testing:** 340 held-out personal calibration frames (20 frames $\times$ 17 letters).
- **Accuracy:**
  - Template Matcher: **41.76%** (Macro F1: 0.3506)
  - KNN: **40.88%** (Macro F1: 0.3571)
  - SVM: **36.76%** (Macro F1: 0.3109)
  - MLP: **23.53%** (Macro F1: 0.1832)
  - Random Forest: **21.47%** (Macro F1: 0.1727)
- **Key Finding:** Models trained exclusively on a single external signer fail to generalize to a new user. Letters like $A, K, L, N, O$ suffered **0.0% recall** due to differences in hand anatomy, camera perspective, and signing position.

### Experiment B: Personal Calibration Data Only
- **Training:** 40 personal calibration samples per letter ($N=800$, 20 letters).
- **Testing:** 20 held-out personal samples per letter ($N=400$).
- **Accuracy:**
  - **Template Matcher:** **100.00%** (Macro F1: 1.0000)
  - **Random Forest:** **100.00%** (Macro F1: 1.0000)
  - **SVM:** **98.75%** (Macro F1: 0.9873)
  - **KNN:** **98.00%** (Macro F1: 0.9796)
  - **MLP:** **96.50%** (Macro F1: 0.9620)
- **Sample Efficiency Ablation (How Many Calibration Templates Are Needed?):**

| Samples / Letter | Template Matcher | KNN | SVM |
| :---: | :---: | :---: | :---: |
| **5 samples** | 95.0% | 82.0% | 76.5% |
| **10 samples** | 99.8% | 85.3% | 79.7% |
| **20 samples** | 100.0% | 89.2% | 84.3% |
| **30 samples** | 100.0% | 94.0% | 92.0% |
| **40 samples** | 100.0% | 98.0% | 98.8% |

- **Key Finding:** As few as **10 samples per letter** (~1.2 seconds of signing) provides **99.8% accuracy** with Template Matching and **85.3%** with KNN. Instance-based template matching excels on small calibration sizes because it does not require fitting complex decision boundaries.

### Experiment C: General + Personal Data Combined
- **Training:** Public dataset (6,800 samples) + Personal calibration (680 samples) = 7,480 samples.
- **Testing:** 340 held-out personal samples (17 shared letters).
- **Accuracy:**
  - **Template Matcher:** **100.00%**
  - **SVM:** **100.00%**
  - **KNN:** **100.00%**
  - **Random Forest:** **100.00%**
  - **MLP:** **100.00%**
- **Key Finding:** Combining both datasets completely resolves the cross-signer transfer failure. The personal samples anchor the decision boundary to the user's specific hand geometry, while the public data provides broader background variation.

### Experiment D: Personal Adaptation (Hybrid Model)
- **Architecture:** Base General SVM (trained on public data) combined with a Personal KNN adapter.
- **Decision Logic:** If the Personal KNN confidence $\ge \tau$, accept the personal prediction; otherwise fall back to General SVM.
- **Results across Threshold $\tau$:**
  - $\tau = 0.3$: Accuracy **98.53%** (100% personal override)
  - $\tau = 0.5$: Accuracy **98.53%** (100% personal override)
  - $\tau = 0.7$: Accuracy **98.24%** (99.1% personal override)
  - $\tau = 0.8$: Accuracy **98.24%** (98.2% personal override)
- **Key Finding:** Lightweight personal adaptation successfully lifts General SVM accuracy from **36.76%** to **98.53%** without retraining the base network.

---

## 7. Answers to Supervisor Questions

1. **Can the system recognize letters using personal templates alone?**  
   **Yes.** Personal calibration achieves **100.00%** accuracy with Template Matching and **98.75%** with SVM.
2. **How does personal-only compare against general-only recognition?**  
   Personal-only (**100.00%**) vastly outperforms general-only zero-shot transfer (**41.76%**). Hand geometry differences make general-only recognition unreliable for unseen signers.
3. **Does combining both data sources improve results?**  
   **Yes.** Combining general and personal data brings accuracy to **100.00%** across all classifiers.
4. **What is the minimum practical calibration data requirement?**  
   Experimental evidence demonstrates that **10 to 20 samples per letter** (1.2 to 2.4 seconds per letter at 8 FPS) is sufficient to reach $\ge 99.8\%$ accuracy with Template Matching and $\ge 85.3\%$ with KNN.
5. **Which letters are most difficult?**  
   Under general zero-shot transfer, $A, K, L, N, O$ suffered 0% recall. Under personal calibration, minor ambiguities occurred between $T$ and $N$ and between $E$ and $I$.
6. **Does personalization reduce confusion between similar letters?**  
   **Yes.** Personalization eliminated cross-signer palm orientation ambiguities, lifting similar-letter recall by over 50 percentage points.
7. **What are the limitations?**  
   Existing personal samples were captured in single continuous bursts. Multi-day evaluation requires recording a second session. Three letters ($H, J, Y$) are absent from the external dataset.

---

## 8. Live Application Integration

### Recognition Backend Architecture
A unified `MLBackend` class (`ml/backend.py`) was integrated into `app.py`:
- **Dual Mode Support:** Supports `BACKEND_TEMPLATE` and `BACKEND_ML` seamlessly.
- **Single Model Load:** The selected model is loaded once at startup or upon settings modification, never inside the video frame loop.
- **Safety Fallback:** If an ML model artifact is missing or corrupt, the system automatically falls back to Template Matching with a user notification.
- **Output Compatibility:** `MLPrediction` is 100% duck-type compatible with `matcher.Prediction`, preserving smoothing (`Smoother`), hold-to-add logic (`HoldToAdd`), and GUI rendering.
- **Teammate Readiness (Task 4):** Exposes `scores` (full class distribution), `margin` (top-1 minus top-2 gap), and `latency_ms` for confidence estimation.

### Settings UI Integration
An option menu ("Recognition Method") was integrated into the Settings dialog (`app.py`), allowing users to toggle between **Template Matching** and **Machine Learning (SVM)** in real time.

---

## 9. Verification and Test Results

The test suite was executed in the project virtual environment:
```bash
python -m pytest tests/test_pipeline.py tests/test_ml.py -v
```
**Results:** **49 passed, 0 failed in 10.23s.**

- 25 Pipeline and tracking tests: **PASSED**
- 24 ML and integration tests: **PASSED**
  - Dataset dimension and finite-value validation
  - Non-leakage split verification
  - Synthetic training fixtures for KNN, SVM, RF, MLP
  - Model serialization roundtrip
  - Missing-hand and zero-vector robustness
  - UI duck-type compatibility and settings persistence

---

## 10. Conclusion & Recommendations

1. **Best Model Recommendation:**  
   - For **general dataset classification**, **SVM (RBF, $C=10$)** is the recommended ML model (99.95% accuracy, 6.85 ms latency).
   - For **ultra-fast real-time systems**, **MLP** is recommended (99.35% accuracy, 0.158 ms latency).
2. **Personal Calibration Recommendation:**  
   - The original **Template Matcher** remains the superior lightweight calibration engine for instant, few-shot personalization (10 samples/letter yields 99.8% accuracy without model fitting).
3. **Hybrid Path:**  
   - The optimal production architecture is the **Hybrid Adaptation Backend (Experiment D)**: a general SVM trained on public data combined with a fast personal template cache for instant user adaptation.

