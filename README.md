# Machine Learning — Practical Session 3

**Topic:** Hypothesis spaces, feature representations and model generalization.

## Dataset
The solution uses the ten observations provided in the TP:

| Student | Study hours | Exam score |
|---|---:|---:|
| S1 | 1 | 42 |
| S2 | 2 | 48 |
| S3 | 3 | 55 |
| S4 | 4 | 59 |
| S5 | 5 | 66 |
| S6 | 6 | 70 |
| S7 | 7 | 78 |
| S8 | 8 | 81 |
| S9 | 9 | 89 |
| S10 | 10 | 91 |

## Requirements
- Python 3.9+
- numpy
- pandas
- matplotlib
- scikit-learn

Install dependencies:

```bash
pip install numpy pandas matplotlib scikit-learn
```

## Run

```bash
python tp3_solution.py
```

The script prints the Part A–D result tables and saves plots (`.png`) and result tables (`.csv`) in the working directory.

## Methodology
- Part A compares linear regression and polynomial regression of degrees 2 and 3.
- Part B compares polynomial degrees 1, 2, 3, 5 and 9 using Leave-One-Out Cross-Validation (LOOCV).
- Part C compares feature degrees 1, 3 and 5, then tests Ridge and Lasso with alpha values 0.01, 0.1, 1 and 10.
- Part D compares linear regression, the best unregularized polynomial degree among 2, 3 and 5, and degree-5 Ridge/Lasso models using their best tested alpha.

For Ridge and Lasso, polynomial feature generation and standardization are placed inside a scikit-learn `Pipeline`. This ensures scaling is learned only from the training portion of each LOOCV fold and avoids data leakage.

**Note:** The dataset contains only ten observations. LOOCV results can be sensitive to individual observations, so conclusions should be interpreted cautiously.
