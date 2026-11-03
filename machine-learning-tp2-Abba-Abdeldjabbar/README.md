# ML TP2 – Standardized PCA

Implementation of standardized PCA from scratch (NumPy/Pandas), compared with Scikit-learn's `PCA`.

- **Data:** grades of 6 students in 4 subjects (Mathematics, Statistics, Programming, Algorithms)
- **Notebook:** `TP-02-ABBA-Abdeldjabbar.ipynb` (run cells in order)

---

## Part A – PCA from Scratch

| Step | Description |
|------|-------------|
| 01 | Load dataset into DataFrame |
| 02 | Dimensions & descriptive statistics |
| 03 | Standardization (z-score) |
| 04 | Correlation matrix |
| 05 | Eigenvalues/eigenvectors of covariance matrix |
| 06 | Sort eigenvalues & reorder eigenvectors |
| 07 | Select top 2 eigenvectors (PC1, PC2) |
| 08 | Project standardized data onto PC1, PC2 |
| 09 | Explained variance ratio & cumulative |

**Result (Part A):** PC1 ≈ 93.5%, PC2 ≈ 6.0% → **99.5% variance retained** with 2 components.

---

## Part B – PCA with Scikit-learn

| Step | Description |
|------|-------------|
| 10 | Apply `sklearn.decomposition.PCA` on standardized data |
| 11 | Reduce to 2 components |
| 12 | Transformed coordinates of the students |
| 13 | Explained variance ratio & cumulative |

## Part C – Comparison & Interpretation

| Step | Description |
|------|-------------|
| 14 | Compare coordinates (manual vs sklearn) |
| 15 | Compare eigenvalues & explained variance ratios |
| 16 | Plot students in 2D with both implementations |
| 17 | Explain differences (sign changes) |
| 18 | Variance retained by the first 2 PCs |

---

## Key Findings

- **PC1 (Learning Level):** overall level of a student — all subjects have the same sign and almost equal weight
- **PC2 (Learning Profile):** Statistics vs Programming/Algorithms (Mathematics ≈ neutral)
- **2D representation** captures 99.5% of the variance
- **Manual PCA = sklearn** (same eigenvalues and ratios; only PC1 is sign-flipped — the subspace is identical)

---

## Requirements

```bash
pip install numpy pandas scikit-learn matplotlib
```
