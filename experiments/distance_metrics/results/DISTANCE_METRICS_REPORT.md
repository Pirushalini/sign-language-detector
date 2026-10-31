# Task 2: Comparative Analysis of Alternative Distance Metrics
## Sign Language Detector (Two-Handed Fingerspelling A–Z)

> [!NOTE]
> Evaluation performed on real hand landmark template datasets under a stratified 80/20 split.

### 1. Executive Summary

- **Evaluated Metrics**: Euclidean, Cosine, Standardized Euclidean, Mahalanobis.
- **Experimental Dataset**: 7456 training samples, 1864 test queries.
- **Highest Accuracy**: **Mahalanobis** (99.89%)
- **Highest Macro F1**: **Mahalanobis** (99.90%)
- **Fastest Inference**: **Cosine** (0.165 ms / prediction)

### 2. Comprehensive Performance Comparison

| Metric | Accuracy | Macro F1 | Macro Precision | Macro Recall | Latency (ms) | Top-2 Margin | Brier Score | ECE |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Euclidean** | 99.84% | 99.86% | 99.86% | 99.86% | 0.281 ms | 0.925 | 0.0183 | 0.0447 |
| **Cosine** | 99.79% | 99.81% | 99.81% | 99.81% | 0.165 ms | 0.041 | 0.8231 | 0.8771 |
| **Seuclidean** | 99.68% | 99.71% | 99.72% | 99.71% | 0.263 ms | 0.994 | 0.0063 | 0.0026 |
| **Mahalanobis** | 99.89% | 99.90% | 99.91% | 99.90% | 0.295 ms | 0.998 | 0.0010 | 0.0011 |

### 3. Detailed Metric Formulation & Mathematical Analysis

#### 1. Euclidean Distance (Baseline)
- **Formula**: $d_E(\mathbf{q}, \mathbf{x}) = \sqrt{\sum_{i=1}^{132} (q_i - x_i)^2}$
- **Behavior**: Standard straight-line distance. Assumes isotropic, uncorrelated landmark noise across all 132 features.
- **Observation**: High-variance fingertips contribute heavily to the distance, which can cause confusion on letters with identical hand shapes but subtle palm/wrist rotations.

#### 2. Cosine Distance
- **Formula**: $d_C(\mathbf{q}, \mathbf{x}) = 1 - \frac{\mathbf{q} \cdot \mathbf{x}}{\|\mathbf{q}\|_2 \|\mathbf{x}\|_2}$
- **Behavior**: Measures directional angular alignment of the feature vectors, discarding magnitude.
- **Safeguards**: Zero vectors (e.g. absent Hand 2) are safely guarded against division by zero ($0/0$).

#### 3. Standardized Euclidean Distance
- **Formula**: $d_{SE}(\mathbf{q}, \mathbf{x}) = \sqrt{\sum_{i=1}^{132} \frac{(q_i - x_i)^2}{\sigma_i^2}}$
- **Behavior**: Normalizes each feature axis by its sample variance $\sigma_i^2$ computed strictly from training templates.
- **Safeguards**: Enforces a variance floor $\sigma_i^2 \ge 10^{-6}$ for fixed coordinates (such as the centered wrist origin $(0, 0, 0)$).

#### 4. Mahalanobis Distance
- **Formula**: $d_M(\mathbf{q}, \mathbf{x}) = \sqrt{(\mathbf{q} - \mathbf{x})^T \mathbf{\Sigma}_{\text{reg}}^{-1} (\mathbf{q} - \mathbf{x})}$
- **Behavior**: Decorrelates anatomical landmark features (e.g., finger joints moving in unison).
- **Safeguards**: Employs Tikhonov ridge and shrinkage regularization $\mathbf{\Sigma}_{\text{reg}} = (1 - \alpha)\mathbf{\Sigma} + \alpha \mathbf{I} + \lambda \mathbf{I}$ with Cholesky whitening factorization.

### 4. Per-Letter Accuracy Highlights

Per-letter breakdown summary (stored in full in `per_letter_recall.csv`):
- **Euclidean** lowest recall letters: I (97.5%), U (98.8%), A (100.0%)
- **Cosine** lowest recall letters: I (97.5%), U (97.5%), A (100.0%)
- **Seuclidean** lowest recall letters: I (96.2%), U (97.5%), T (98.8%)
- **Mahalanobis** lowest recall letters: O (97.5%), A (100.0%), B (100.0%)

### 5. Generated Artifacts in `experiments/distance_metrics/results/`

- `metrics_comparison.csv`: Summary metrics table across all 4 distance formulations.
- `per_letter_recall.csv`: Matrix of recall values for all letters A–Z.
- `prediction_scores.csv`: Per-query classification decisions, confidences, and top-2 margins.
- `comparison_barchart.png`: 4-panel visual comparison (Accuracy, F1, Margin, Latency).
- `confusion_matrix_<metric>.png`: Full $26 \times 26$ confusion matrices for each metric.
