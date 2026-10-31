# Task 2 Final Report: Comparative Analysis of Alternative Distance Metrics
**Project**: Sign Language Detector (Two-Handed Fingerspelling A–Z)  
**Branch**: `feature/distance-metrics`  
**Dataset**: 9,320 Real Hand Landmark Vectors across 26 Classes (A–Z)  
**Author**: Sandari (Internship Project)

---

## 1. Executive Summary & Objective

In this project, the **Sign Language Detector** recognizes the British two-handed fingerspelling alphabet from a webcam video stream using **non-parametric template matching** (instance-based classification). Up to two hands are tracked via Google MediaPipe Hands and converted into a fixed-length **132-dimensional feature vector**.

### The Core Problem with Standard Euclidean Distance
The original system matches live query vectors against stored templates using standard **Euclidean Distance**. However, standard Euclidean distance makes two major assumptions that violate hand anatomy:
1. **Equal Variance**: It assumes all 132 features fluctuate by the same amount. In reality, fingertips move across large physical ranges during signing, while base knuckles remain relatively steady. In Euclidean space, noisy fingertip movement dominates subtle knuckle orientations.
2. **Feature Independence (Zero Covariance)**: It assumes all landmark coordinates move independently. In reality, human hand joints are biomechanically coupled—when a finger bends, its MCP, PIP, and DIP joints rotate together along correlated trajectories. Euclidean distance treats these as three independent pieces of evidence, artificially tripling their weight.

### The Objective of Task 2
To formulate, implement, and scientifically compare **four alternative distance metrics**:
1. **Euclidean Distance** (Baseline)
2. **Cosine Distance** (Directional / Angular)
3. **Standardized Euclidean Distance** (Variance-Normalized)
4. **Mahalanobis Distance** (Covariance-Decorrelated)

All metrics were benchmarked under **strictly identical experimental splits**, with rigorous numerical safeguards, metric-specific temperature calibration, and a transparent methodological audit.

---

## 2. Mathematical Formulations & Anatomical Intuition

```
                        The 4 Metric Geometries
                        
   1. Euclidean Distance              2. Cosine Distance
      Straight-line distance             Angular orientation
             x (Template)                       x (Template)
            /                                  /  θ
           / d                                / ──┐
          /                                  /    │
         q (Query)                          q─────┴─────
  
   3. Standardized Euclidean          4. Mahalanobis Distance
      Normalizes by feature variance     Rotates & scales along joint
      (Downweights noisy fingertips)     correlation ellipses
             y ▲                                y ▲   .-'""'-.
               │    (ellipse)                     │ .'        '.
               │   .-'""'-.                       │/  joint     \
               │  /        \                      │\  correlations
               └──────────────► x                 └──────────────► x
```

### 1. Euclidean Distance (Baseline)
$$d_E(\mathbf{q}, \mathbf{x}) = \sqrt{\sum_{i=1}^{132} (q_i - x_i)^2} = \|\mathbf{q} - \mathbf{x}\|_2$$
- **Geometry**: Direct straight-line distance in $\mathbb{R}^{132}$.
- **Anatomical Characteristic**: Isotropic (spherical contour). High-variance fingertips contribute heavily to the distance, which can cause confusion on letters with identical hand shapes but subtle palm/wrist rotations.

### 2. Cosine Distance
$$d_C(\mathbf{q}, \mathbf{x}) = 1.0 - \frac{\mathbf{q} \cdot \mathbf{x}}{\|\mathbf{q}\|_2 \|\mathbf{x}\|_2}$$
- **Geometry**: Measures the cosine of the angle $\theta$ between query and template vectors, discarding vector magnitudes.
- **Anatomical Characteristic**: Compares the directional alignment of finger bones regardless of overall hand size or proximity to the camera lens.

### 3. Standardized Euclidean Distance (SEuclidean)
$$d_{SE}(\mathbf{q}, \mathbf{x}) = \sqrt{\sum_{i=1}^{132} \frac{(q_i - x_i)^2}{\sigma_i^2}}$$
where $\sigma_i^2$ is the sample variance of feature $i$ computed strictly from training templates.
- **Geometry**: Axis-aligned ellipsoidal distance. Each coordinate axis is scaled by $1/\sigma_i$.
- **Anatomical Characteristic**: Downweights high-variance, jittery fingertip points and amplifies low-variance, stable reference landmarks (such as wrist and palm knuckles).

### 4. Mahalanobis Distance
$$d_M(\mathbf{q}, \mathbf{x}) = \sqrt{(\mathbf{q} - \mathbf{x})^T \mathbf{\Sigma}_{\text{reg}}^{-1} (\mathbf{q} - \mathbf{x})}$$
where $\mathbf{\Sigma}_{\text{reg}} \in \mathbb{R}^{132 \times 132}$ is the regularized covariance matrix estimated from training templates.
- **Geometry**: Rotated ellipsoidal distance aligned with the principal axes of joint co-variation.
- **Anatomical Characteristic**: Directly accounts for anatomical joint coupling. If moving an index finger shifts coordinates $x_5, x_6, x_7, x_8$ simultaneously, Mahalanobis distance penalizes them as a single cohesive movement rather than quadruple-counting them.

---

## 3. Implementation Details & Numerical Safeguards

All distance metrics were implemented from scratch in [experiments/distance_metrics/metrics.py](file:///c:/Users/USER/Desktop/sign_vision/sign-language-detector/experiments/distance_metrics/metrics.py) with strict numerical stability guarantees:

1. **Cosine Zero-Vector Handling**:
   - In two-handed signing, signs with only 1 hand (e.g., 'C') have all zeros in the Hand 2 slice `[63:126]`. If a query vector has zero norm, direct division causes `0/0 = NaN`.
   - *Safeguard*: Vectors with norm $< 10^{-9}$ are clamped. Distance between two zero vectors is defined as $0.0$; distance between a zero and non-zero vector is defined as $1.0$. Distances are clipped to $[0.0, 2.0]$.
2. **Standardized Euclidean Variance Floor**:
   - Coordinates permanently centered at zero (such as the wrist landmark at $(0, 0, 0)$) have zero variance across all training frames ($\sigma_i^2 = 0$).
   - *Safeguard*: A variance floor is enforced: $\sigma_{i, \text{safe}}^2 = \max(\sigma_i^2, 10^{-6})$.
3. **Mahalanobis Regularization & Cholesky Whitening**:
   - In 132-dimensional space, sample covariance matrices $\mathbf{\Sigma}$ are frequently ill-conditioned or singular due to collinear joint geometry.
   - *Safeguards*:
     1. **Shrinkage & Ridge Regularization**:
        $$\mathbf{\Sigma}_{\text{reg}} = (1 - \alpha)\mathbf{\Sigma} + \alpha \frac{\text{Tr}(\mathbf{\Sigma})}{D}\mathbf{I} + \lambda \mathbf{I}$$
     2. **Cholesky Factorization**: We decompose $\mathbf{\Sigma}_{\text{reg}} = \mathbf{L}\mathbf{L}^T$ and compute the whitening matrix $\mathbf{W} = \mathbf{L}^{-T}$. Mahalanobis distance between query $\mathbf{q}$ and template $\mathbf{x}$ reduces to fast Euclidean distance in whitened space:
        $$d_M(\mathbf{q}, \mathbf{x}) = \|(\mathbf{q} - \mathbf{x})\mathbf{W}\|_2$$
        This guarantees strict positive definiteness, avoids matrix inversion instability, and reduces computational complexity from $O(M \cdot N \cdot D^2)$ to $O((M + N)D^2 + M \cdot N \cdot D)$.

---

## 4. Experimental Dataset Description

All evaluations were executed on the **real, combined project dataset** (zero synthetic data used for final evaluation):

| Data Source | Letters Covered | Sample Count | Percentage of Dataset |
|---|:---:|:---:|:---:|
| **External BSL Landmark Dataset** (Kaggle) | 23 Letters (A–G, I, K–X, Z) | 9,200 samples ($400$ per letter) | 98.71% |
| **Personal Webcam Calibration** | 3 Letters (H, J, Y) | 120 samples ($40$ per letter) | 1.29% |
| **Total Combined Dataset** | **All 26 Letters (A–Z)** | **9,320 samples** | **100.00%** |

---

## 5. Experimental Results

### Protocol 1: Random Stratified Split (80% Train, 20% Test)
- **Training samples**: 7,456
- **Test queries**: 1,864
- **Evaluated under strictly identical train/test splits**:

| Metric | Accuracy | Macro F1 | Macro Precision | Macro Recall | Latency (ms) | Top-2 Margin | Brier Score | ECE | Correct / Total | Errors |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Mahalanobis** | **99.89%** | **99.90%** | **99.91%** | **99.90%** | 0.295 ms | **0.998** | **0.0010** | **0.0011** | **1,862 / 1,864** | **2** |
| **Euclidean** (Baseline) | 99.84% | 99.86% | 99.86% | 99.86% | 0.281 ms | 0.925 | 0.0183 | 0.0447 | 1,861 / 1,864 | 3 |
| **Cosine** | 99.79% | 99.81% | 99.81% | 99.81% | **0.165 ms** | 0.041 | 0.8231 | 0.8771 | 1,860 / 1,864 | 4 |
| **SEuclidean** | 99.68% | 99.71% | 99.72% | 99.71% | 0.263 ms | 0.994 | 0.0063 | 0.0026 | 1,858 / 1,864 | 6 |

---

### Protocol 2: Block-Based Split (Contiguous Chunk Holdout)
To test performance when temporal blocks are held out rather than randomly scattered frames, each letter class was split into 5 contiguous blocks. Block 5 (20%) was used as the test set, and Blocks 1–4 (80%) were used for training:

| Metric | Random Split Accuracy | Random Macro F1 | Block Split Accuracy | Block Macro F1 | Latency (ms) |
|:---|:---:|:---:|:---:|:---:|:---:|
| **Mahalanobis** | **99.89%** | **99.90%** | **99.84%** | **99.15%** | 0.194 ms |
| **Cosine** | 99.79% | 99.81% | 99.73% | 99.26% | **0.108 ms** |
| **Euclidean** | 99.84% | 99.86% | 99.68% | 99.21% | 0.219 ms |
| **SEuclidean** | 99.68% | 99.71% | 99.46% | 99.02% | 0.193 ms |

**Takeaway**: Under the stricter block split, accuracy drops slightly across all metrics, reflecting a tougher decision boundary. **Mahalanobis Distance maintained the highest accuracy across both protocols ($99.89\%$ and $99.84\%$).**

---

### Protocol 3: Metric-Specific Confidence Calibration (70% Train, 10% Val, 20% Test)
To fairly evaluate probabilistic confidence and resolve calibration anomalies, a 3-way split was applied. The temperature parameter $\tau^*$ was optimized exclusively on the **10% Validation set** (minimizing NLL loss), and evaluated on the **20% Test set**:

| Metric | Baseline $\tau$ | Optimal $\tau^*$ | Accuracy | ECE (Before) | **ECE (After)** | Brier (Before) | **Brier (After)** | Mean Confidence |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Cosine** | 0.12 | **0.00032** | 99.79% | 0.8770 | **0.0021** | 0.8232 | **0.0030** | 12.08% $\to$ **99.90%** |
| **Euclidean** | 0.12 | **0.01689** | 99.84% | 0.0462 | **0.0018** | 0.0192 | **0.0032** | 95.22% $\to$ **99.90%** |
| **Mahalanobis** | 0.12 | **0.09598** | **99.89%** | 0.0014 | **0.0015** | 0.0023 | **0.0023** | 99.90% $\to$ **99.92%** |
| **SEuclidean** | 0.12 | **0.17044** | 99.41% | 0.0027 | **0.0057** | 0.0079 | **0.0080** | 99.61% $\to$ **99.24%** |

#### Resolution of the Cosine Calibration Mystery:
- Under baseline temperature ($\tau = 0.12$), Cosine distance had an apparent calibration failure ($\text{ECE} = 0.8770$), with mean confidence of only $12.08\%$.
- *Reason*: Cosine distances are bounded in $[0.0, 2.0]$, meaning distance gaps between classes are tiny ($\approx 0.015$). When divided by $\tau = 0.12$, the softmax distribution becomes flat.
- When tuned on validation data, the optimal temperature dropped to **$\tau^* = 0.00032$**.
- This dropped ECE from **$0.8770$ down to $0.0021$ ($99.8\%$ reduction)** and Brier score from **$0.8232$ to $0.0030$**, with **accuracy perfectly preserved at $99.79\%$**.

---

## 6. Confusion Matrix & Per-Letter Recall Findings

### The 1.25% Recall Step Proof
Each external letter has 80 test samples ($400 \times 0.20 = 80$). Each misclassified test sample produces a step reduction of:
$$\frac{1}{80} = 0.0125 = \mathbf{1.25\%}$$
- 0 errors $\to 80/80 = \mathbf{100.00\%}$
- 1 error $\to 79/80 = \mathbf{98.75\%}$
- 2 errors $\to 78/80 = \mathbf{97.50\%}$
- 3 errors $\to 77/80 = \mathbf{96.25\%}$

### Letters Exhibiting Misclassifications
Out of 26 letters, **22 letters achieved a perfect 100% recall across all metrics**. All errors were concentrated in four letters:

| Letter | Euclidean | Cosine | Standardized Euclidean | Mahalanobis |
|:---:|:---:|:---:|:---:|:---:|
| **I** | 97.50% (2 errors) | 97.50% (2 errors) | 96.25% (3 errors) | **100.00% (0 errors)** |
| **O** | **100.00% (0 errors)** | **100.00% (0 errors)** | **100.00% (0 errors)** | 97.50% (2 errors) |
| **T** | **100.00% (0 errors)** | **100.00% (0 errors)** | 98.75% (1 error) | **100.00% (0 errors)** |
| **U** | 98.75% (1 error) | 97.50% (2 errors) | 97.50% (2 errors) | **100.00% (0 errors)** |

### Anatomical Explanation of Confused Letter Pairs
1. **The Vowels (I, O, U, E)**:
   - In British Sign Language fingerspelling, vowels are signed by touching the non-dominant fingertips (Thumb = A, Index = E, Middle = I, Ring = O, Pinky = U).
   - Because adjacent fingertips are physically separated by merely 1 to 2 centimeters, camera noise and finger contact cause subtle misalignments between **I (middle finger)**, **O (ring finger)**, and **U (pinky finger)**.
2. **Consonants T and N**:
   - For `T`, the dominant index touches the side of the non-dominant palm.
   - For `N`, the dominant index and middle fingers touch the palm face.
   - Their overall hand positioning is nearly identical, making them a classic confusion pair.

---

## 7. Inference Time & Real-Time Suitability

All four distance metrics were benchmarked for average latency per single-frame query:
- **Cosine Distance**: **0.165 ms** (Fastest, capable of $\approx 6,000$ FPS)
- **Standardized Euclidean**: **0.263 ms** ($\approx 3,800$ FPS)
- **Euclidean Distance**: **0.281 ms** ($\approx 3,500$ FPS)
- **Mahalanobis Distance**: **0.295 ms** ($\approx 3,300$ FPS)

**Verdict**: Any of the four metrics easily runs within a standard 33 ms video frame budget ($30$ FPS webcam capture), consuming less than **$1\%$ of available frame processing time**.

---

## 8. Scientific Limitations & Data Leakage Discussion

> [!IMPORTANT]
> **Key Distinctions between Measured Results and Real-World Performance**:
> 1. **Temporal Autocorrelation (Near-Identical Video Frames)**:
>    - The Kaggle dataset consists of continuous video recordings captured at 20–30 FPS.
>    - In a random stratified split, frame $t$ (e.g. frame 142) can appear in training, while frame $t+1$ (frame 143, captured 33 ms later) appears in testing.
>    - Because adjacent frames are near-identical, the $>99.8\%$ accuracy reflects **temporal interpolation within the same recording**, rather than generalization across completely new signers.
> 2. **Absence of Temporal & Signer Metadata**:
>    - Our technical audit confirmed that the Kaggle CSV files contain **no timestamps, no signer IDs, and no session numbers**.
>    - Furthermore, `import_dataset.py` randomly sampled 400 rows per class (`random.sample(rows, 400)`).
>    - Therefore, block splitting within the template array cannot definitively guarantee that consecutive video frames were isolated.
> 3. **Mixing External and Personal Datasets**:
>    - Letters A–G, I, K–X, Z came from the Kaggle contributor's camera and hands.
>    - Letters H, J, Y came from personal webcam calibration.
>    - While this provided templates for all 26 letters, it created a hybrid dataset with two different lighting and camera environments.

---

## 9. Final Conclusion & Recommendations

### Best Performing Metric: **Mahalanobis Distance**
1. **Accuracy Champion**: Achieved the highest accuracy across both evaluation protocols (**99.89%** random split, **99.84%** block split).
2. **Superior Discrimination**: Produced a top-2 probability margin of **0.998**, decisively separating correct signs from competitors.
3. **Anatomical Soundness**: By regularizing the covariance matrix $\mathbf{\Sigma}_{\text{reg}}$, it decorrelates coupled hand joints and eliminates vowel confusions between letters I and U.
4. **Real-Time Efficiency**: Thanks to the Cholesky whitening transform, latency remained negligible (**0.295 ms**).

### Secondary Recommendation: **Cosine Distance for Edge / Low-Power Compute**
- If deploying to low-power mobile or microcontroller hardware without floating-point matrix units, **Cosine Distance** is optimal ($1.8\times$ faster at 0.165 ms and 99.79% accuracy), provided the calibrated temperature $\tau^* \approx 0.0003$ is applied.

---

## 10. Codebase Structure & How to Reproduce

### File Layout
```
experiments/distance_metrics/
├── __init__.py                     # Package interface
├── metrics.py                      # 4 Distance implementations & safeguards
├── matcher.py                      # DistanceTemplateMatcher (k-NN, user weights)
├── evaluator.py                    # Evaluator engine (accuracy, F1, latency, ECE, plots)
├── calibration.py                  # Metric-specific temperature scaling (TASK A)
├── run_experiments.py              # Baseline experiment runner
├── rigorous_evaluation.py          # 3-way split & protocol comparison runner (TASK B)
├── FINAL_TASK2_REPORT.md           # Comprehensive final report
└── results/
    ├── metrics_comparison.csv      # Baseline comparison metrics
    ├── per_letter_recall.csv       # Per-letter recall matrix
    ├── prediction_scores.csv       # Query-level classification scores
    ├── calibration_comparison.csv  # Before/after calibration comparison
    ├── protocol_comparison.csv     # Random vs Block split comparison
    ├── DISTANCE_METRICS_REPORT.md  # Detailed initial report
    ├── RIGOROUS_EVALUATION_REPORT.md # Rigorous audit report
    ├── comparison_barchart.png     # 4-panel visual chart
    └── confusion_matrix_*.png      # 4 Confusion matrix plots
```

### Commands to Reproduce Experiments & Tests
```powershell
# 1. Run full unit test suite (28 tests across all modules)
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe -m pytest tests/test_distance_metrics.py tests/test_distance_evaluator.py tests/test_calibration_and_protocols.py -v

# 2. Run baseline evaluation
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe -m experiments.distance_metrics.run_experiments

# 3. Run rigorous 3-way calibration and protocol comparison
$env:PYTHONUTF8="1"; .\.venv\Scripts\python.exe -m experiments.distance_metrics.rigorous_evaluation
```
