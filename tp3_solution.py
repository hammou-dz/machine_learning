"""
Machine Learning - Practical Session 3
Hypothesis spaces, feature representations and model generalization

Dataset: study hours (x) and exam scores (y) for 10 students.
Run: python tp3_solution.py

Dependencies:
    pip install numpy pandas matplotlib scikit-learn
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import LeaveOneOut, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score


# ---------------------------------------------------------------------
# 0. Dataset exploration
# ---------------------------------------------------------------------
students = [f"S{i}" for i in range(1, 11)]
X = np.arange(1, 11, dtype=float).reshape(-1, 1)
y = np.array([42, 48, 55, 59, 66, 70, 78, 81, 89, 91], dtype=float)

data = pd.DataFrame({
    "Student": students,
    "Study Hours (x)": X.ravel(),
    "Exam Score (y)": y
})

print("\\nDATASET")
print(data.to_string(index=False))
print("\\nDescriptive statistics")
print(data[["Study Hours (x)", "Exam Score (y)"]].describe())

plt.figure(figsize=(7, 5))
plt.scatter(X, y, color="black", label="Observed data", zorder=3)
plt.xlabel("Study hours")
plt.ylabel("Exam score")
plt.title("Study hours vs. exam scores")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("part_A_dataset.png", dpi=160)
plt.show()


def polynomial_model(degree, estimator=None):
    """Polynomial features followed by a regression estimator."""
    if estimator is None:
        estimator = LinearRegression()
    return Pipeline([
        ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
        ("regression", estimator)
    ])


def training_rmse(model, X_train, y_train):
    """RMSE on the same observations used to fit the model."""
    model.fit(X_train, y_train)
    return float(np.sqrt(mean_squared_error(y_train, model.predict(X_train))))


def loocv_rmse(model, X_data, y_data):
    """LOOCV RMSE calculated from negative mean squared error."""
    scores = cross_val_score(
        model,
        X_data,
        y_data,
        cv=LeaveOneOut(),
        scoring="neg_mean_squared_error"
    )
    return float(np.sqrt(-scores.mean()))


# ---------------------------------------------------------------------
# Part A - Linear and polynomial regression (degrees 2 and 3)
# ---------------------------------------------------------------------
models_A = {
    "Linear Regression": Pipeline([
        ("regression", LinearRegression())
    ]),
    "Polynomial Regression (degree 2)": polynomial_model(2),
    "Polynomial Regression (degree 3)": polynomial_model(3),
}

rows_A = []
x_plot = np.linspace(X.min(), X.max(), 300).reshape(-1, 1)

plt.figure(figsize=(8, 5))
plt.scatter(X, y, color="black", label="Observed data", zorder=4)

for name, model in models_A.items():
    model.fit(X, y)
    predictions = model.predict(X)
    rmse = np.sqrt(mean_squared_error(y, predictions))
    r2 = r2_score(y, predictions)
    rows_A.append({
        "Model": name,
        "Training RMSE": rmse,
        "Training R2": r2
    })
    plt.plot(x_plot, model.predict(x_plot), label=name)

results_A = pd.DataFrame(rows_A)
print("\\nPART A - Training performance")
print(results_A.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

plt.xlabel("Study hours")
plt.ylabel("Exam score")
plt.title("Part A: Regression model fits")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("part_A_model_comparison.png", dpi=160)
plt.show()


# ---------------------------------------------------------------------
# Part B - Model complexity and generalization
# ---------------------------------------------------------------------
degrees = [1, 2, 3, 5, 9]
rows_B = []

for degree in degrees:
    model = polynomial_model(degree)
    rows_B.append({
        "Polynomial Degree": degree,
        "Training RMSE": training_rmse(model, X, y),
        "LOOCV RMSE": loocv_rmse(polynomial_model(degree), X, y)
    })

results_B = pd.DataFrame(rows_B)
print("\\nPART B - Polynomial degree and generalization")
print(results_B.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

plt.figure(figsize=(7, 5))
plt.plot(results_B["Polynomial Degree"], results_B["LOOCV RMSE"],
         marker="o", label="LOOCV RMSE")
plt.xlabel("Polynomial degree")
plt.ylabel("LOOCV RMSE")
plt.title("Part B: Generalization vs. polynomial degree")
plt.xticks(degrees)
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("part_B_loocv_by_degree.png", dpi=160)
plt.show()

best_degree_B = int(results_B.loc[results_B["LOOCV RMSE"].idxmin(),
                                  "Polynomial Degree"])
print(f"Lowest LOOCV RMSE in Part B: polynomial degree {best_degree_B}.")


# ---------------------------------------------------------------------
# Part C1 - Feature representations
# ---------------------------------------------------------------------
feature_degrees = [1, 3, 5]
rows_C1 = []

for degree in feature_degrees:
    model = polynomial_model(degree)
    rows_C1.append({
        "Feature representation": f"Polynomial degree {degree}",
        "Feature degree": degree,
        "LOOCV RMSE": loocv_rmse(model, X, y)
    })

results_C1 = pd.DataFrame(rows_C1)
print("\\nPART C1 - Feature representation comparison")
print(results_C1.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ---------------------------------------------------------------------
# Part C2 - Ridge and Lasso regularization on degree-5 features
# Scaling is inside the Pipeline, so it is refitted within each CV fold.
# ---------------------------------------------------------------------
alphas = [0.01, 0.1, 1.0, 10.0]
rows_regularization = []

for method, estimator_class in [("Ridge", Ridge), ("Lasso", Lasso)]:
    for alpha in alphas:
        model = Pipeline([
            ("poly", PolynomialFeatures(degree=5, include_bias=False)),
            ("scaler", StandardScaler()),
            ("regression", estimator_class(alpha=alpha, max_iter=100000))
        ])
        rows_regularization.append({
            "Model": method,
            "Feature degree": 5,
            "Alpha": alpha,
            "LOOCV RMSE": loocv_rmse(model, X, y)
        })

results_regularization = pd.DataFrame(rows_regularization)
print("\\nPART C2 - Regularization strength comparison")
print(results_regularization.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

best_regularized = (
    results_regularization.sort_values("LOOCV RMSE")
    .groupby("Model", as_index=False)
    .first()
)
print("\\nBest alpha for each regularization method")
print(best_regularized.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# Required Part C comparison table:
# degree-1 linear, degree-3 polynomial, degree-5 polynomial,
# degree-5 Ridge and degree-5 Lasso (best alpha selected by LOOCV).
best_alpha = {
    row["Model"]: float(row["Alpha"])
    for _, row in best_regularized.iterrows()
}
ridge_best = Pipeline([
    ("poly", PolynomialFeatures(degree=5, include_bias=False)),
    ("scaler", StandardScaler()),
    ("regression", Ridge(alpha=best_alpha["Ridge"]))
])
lasso_best = Pipeline([
    ("poly", PolynomialFeatures(degree=5, include_bias=False)),
    ("scaler", StandardScaler()),
    ("regression", Lasso(alpha=best_alpha["Lasso"], max_iter=100000))
])

models_C = {
    "Linear Regression": polynomial_model(1),
    "Polynomial Regression (degree 3)": polynomial_model(3),
    "Polynomial Regression (degree 5)": polynomial_model(5),
    "Ridge Regression (degree 5)": ridge_best,
    "Lasso Regression (degree 5)": lasso_best,
}

rows_C = []
for name, model in models_C.items():
    rows_C.append({
        "Model": name,
        "Feature Degree": 1 if name == "Linear Regression" else
                          (3 if "degree 3" in name else 5),
        "Regularization": "L2" if name.startswith("Ridge") else
                          ("L1" if name.startswith("Lasso") else "None"),
        "LOOCV RMSE": loocv_rmse(model, X, y)
    })

results_C = pd.DataFrame(rows_C)
print("\\nPART C - Required comparison table")
print(results_C.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


# ---------------------------------------------------------------------
# Part D - Final model comparison
# Select the best unregularized polynomial degree among degrees 2, 3, 5
# using LOOCV. Select Ridge/Lasso alpha from the tested values above.
# ---------------------------------------------------------------------
candidate_poly_degrees = [2, 3, 5]
best_poly_degree = min(
    candidate_poly_degrees,
    key=lambda d: loocv_rmse(polynomial_model(d), X, y)
)

final_models = {
    "Linear Regression": polynomial_model(1),
    f"Polynomial Regression (degree {best_poly_degree})":
        polynomial_model(best_poly_degree),
    "Ridge Regression (degree 5)": ridge_best,
    "Lasso Regression (degree 5)": lasso_best,
}

rows_D = []
plt.figure(figsize=(9, 6))
plt.scatter(X, y, color="black", label="Observed data", zorder=5)

for name, model in final_models.items():
    model.fit(X, y)
    pred_train = model.predict(X)
    rows_D.append({
        "Model": name,
        "Training RMSE": np.sqrt(mean_squared_error(y, pred_train)),
        "LOOCV RMSE": loocv_rmse(model, X, y)
    })
    plt.plot(x_plot, model.predict(x_plot), label=name)

results_D = pd.DataFrame(rows_D).sort_values("LOOCV RMSE")
print("\\nPART D - Final model comparison")
print(results_D.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

plt.xlabel("Study hours")
plt.ylabel("Exam score")
plt.title("Part D: Final model comparison")
plt.grid(alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig("part_D_final_comparison.png", dpi=160)
plt.show()

winner = results_D.iloc[0]
print("\\nFINAL INTERPRETATION")
print(f"Lowest LOOCV RMSE: {winner['Model']} "
      f"({winner['LOOCV RMSE']:.4f}).")
print(
    "A lower training RMSE means a closer fit to the observed training data. "
    "LOOCV RMSE estimates performance on unseen observations. "
    "The model with the lowest LOOCV RMSE is the best generalization result "
    "among the models compared here; with only ten observations, this estimate "
    "should be interpreted cautiously."
)

results_A.to_csv("part_A_results.csv", index=False)
results_B.to_csv("part_B_results.csv", index=False)
results_C1.to_csv("part_C_feature_results.csv", index=False)
results_regularization.to_csv("part_C_regularization_results.csv", index=False)
results_C.to_csv("part_C_comparison_results.csv", index=False)
results_D.to_csv("part_D_final_results.csv", index=False)

print("\\nFinished. CSV result tables and PNG plots have been saved in the current folder.")
