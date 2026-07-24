"""
RANDOM FOREST FOR REGRESSION — A LEARNING WALKTHROUGH
======================================================

THE BIG IDEA
------------
Classification predicts a CLASS (0/1, benign/malignant).
Regression predicts a CONTINUOUS NUMBER, such as a house price.

A Random Forest Regressor is an ensemble of many decision trees:

    prediction = AVERAGE of all the trees' predicted numbers

(For classification the forest takes a majority VOTE; for regression it
takes the MEAN. That is the only real conceptual change.)

HOW IT WORKS
------------
1. Grow many decision trees (e.g. 300).
2. Make each tree different using two kinds of randomness:
   - Bagging: each tree trains on a random sample of rows (with replacement).
   - Random features: each split considers only a random subset of features.
3. To predict, ask every tree for its number and average the answers.

Because the trees make different errors, averaging smooths them out, giving
a more accurate and stable prediction than any single tree.

WHY NO FEATURE SCALING
----------------------
A tree split only asks "is this feature above or below a threshold", so the
units/magnitude of a feature do not matter. No StandardScaler needed.

DATASET USED
------------
scikit-learn's built-in Diabetes dataset. It ships inside scikit-learn, so it
needs no download and works fully offline.
Each row is one patient (442 total, 10 numeric features such as age, BMI,
blood pressure, and blood serum measurements).
The target is a continuous score of disease progression one year after
baseline (roughly 25 to 346). Higher = more progression.
The features are already mean-centered and scaled by scikit-learn.
"""

# -----------------------------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------------------------

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# The dataset (bundled inside scikit-learn, so no download is needed).
from sklearn.datasets import load_diabetes

# Splits data into training and testing portions.
from sklearn.model_selection import train_test_split

# The regression algorithm we are learning.
from sklearn.ensemble import RandomForestRegressor

# Regression evaluation metrics.
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)


def print_section(title):
    """Print a section heading so the console output is easy to read."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# -----------------------------------------------------------------------------
# 2. LOAD THE DATASET
# -----------------------------------------------------------------------------

print_section("1. LOAD DATA")

diabetes = load_diabetes()

# X = feature matrix (one row per patient, 10 numeric columns).
# y = target (disease progression score one year after baseline).
X = diabetes.data
y = diabetes.target
feature_names = diabetes.feature_names

print(f"Number of samples/rows: {X.shape[0]}")
print(f"Number of features/columns: {X.shape[1]}")
print(f"Features: {list(feature_names)}")
print("Target: disease progression score (continuous, roughly 25 to 346)")

df_preview = pd.DataFrame(X, columns=feature_names)
df_preview["Progression"] = y
print("\nFirst 5 rows:")
print(df_preview.head())

print("\nTarget summary (disease progression score):")
print(df_preview["Progression"].describe())


# -----------------------------------------------------------------------------
# 3. TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------

print_section("2. SPLIT DATA INTO TRAINING AND TESTING SETS")

# No stratify here: the target is a continuous number, not a class.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
)

print(f"Training rows: {X_train.shape[0]}")
print(f"Testing rows : {X_test.shape[0]}")
print("- Train set: used to grow the trees.")
print("- Test set : unseen data used to check whether the model generalizes.")


# -----------------------------------------------------------------------------
# 4. CREATE AND TRAIN THE MODEL
# -----------------------------------------------------------------------------

print_section("3. CREATE AND TRAIN THE RANDOM FOREST REGRESSOR")

# n_estimators : number of trees (more = stabler, slower).
# max_depth    : how deep each tree can grow (None = grow until leaves are pure).
# max_features : features considered per split (adds randomness).
# n_jobs=-1    : use all CPU cores to train trees in parallel.
model = RandomForestRegressor(
    n_estimators=300,
    max_depth=None,
    max_features=1.0,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

print("Model training completed.")
print(model)
print(f"\nNumber of trees in the forest: {len(model.estimators_)}")


# -----------------------------------------------------------------------------
# 5. FEATURE IMPORTANCES
# -----------------------------------------------------------------------------

print_section("4. WHAT THE MODEL LEARNED: FEATURE IMPORTANCES")

# No coefficients. Instead: how much each feature helped reduce prediction
# error across all splits. Values are >= 0 and sum to 1.0. No direction/sign.
importances = model.feature_importances_

importance_table = pd.DataFrame(
    {"feature": feature_names, "importance": importances}
).sort_values("importance", ascending=False)

print(f"Importances sum to {importances.sum():.4f} (always 1.0).")
print("\nFeatures ranked by importance:")
print(importance_table.to_string(index=False))
print("\nHigher = more useful for splitting. It does NOT tell you direction.")


# -----------------------------------------------------------------------------
# 6. MAKE PREDICTIONS
# -----------------------------------------------------------------------------

print_section("5. MAKE PREDICTIONS ON TEST DATA")

# predict() returns a continuous number (the average of all trees).
y_pred = model.predict(X_test)

prediction_table = pd.DataFrame(
    {
        "actual_value": y_test,
        "predicted_value": y_pred,
        "error": y_test - y_pred,
    }
)

print("First 10 predictions (disease progression score):")
print(prediction_table.head(10).to_string(index=False))


# -----------------------------------------------------------------------------
# 7. PROVE THE PREDICTION IS AN AVERAGE OF THE TREES
# -----------------------------------------------------------------------------

print_section("6. HOW IT WORKS: PREDICTION = AVERAGE OF ALL TREES")

# Take the first test row and ask every individual tree for its number.
one_sample = X_test[0:1]
tree_predictions = np.array([tree.predict(one_sample)[0] for tree in model.estimators_])

manual_average = tree_predictions.mean()
sklearn_prediction = model.predict(one_sample)[0]

print(f"Actual value        : {y_test[0]:.4f}")
print(f"Number of trees     : {len(tree_predictions)}")
print(f"Min tree prediction : {tree_predictions.min():.4f}")
print(f"Max tree prediction : {tree_predictions.max():.4f}")
print(f"Manual average      : {manual_average:.4f}")
print(f"sklearn prediction  : {sklearn_prediction:.4f}")
print("\nThe forest's answer is simply the mean of all the trees' answers.")


# -----------------------------------------------------------------------------
# 8. EVALUATE THE MODEL
# -----------------------------------------------------------------------------

print_section("7. MODEL PERFORMANCE METRICS")

mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f"MAE  (Mean Absolute Error) : {mae:.4f}  (avg miss in progression points)")
print(f"MSE  (Mean Squared Error)  : {mse:.4f}")
print(f"RMSE (Root Mean Sq. Error) : {rmse:.4f}  (in progression points)")
print(f"R2   (coefficient of det.) : {r2:.4f}")

print("\nMetric meaning:")
print("- MAE : average size of the error, in the same units as the target.")
print("- MSE : average of squared errors; punishes big misses harder.")
print("- RMSE: square root of MSE, back in target units; easy to interpret.")
print("- R2  : fraction of the target's variance explained. 1.0 = perfect,")
print("        0.0 = no better than always predicting the mean.")


# -----------------------------------------------------------------------------
# 9. VISUALIZE RESULTS
# -----------------------------------------------------------------------------

print_section("8. CREATE VISUALIZATIONS")

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Random Forest Regression Results", fontsize=14, fontweight="bold")

# PANEL 1: Predicted vs Actual. Perfect predictions lie on the diagonal.
ax = axes[0]
ax.scatter(y_test, y_pred, alpha=0.3, edgecolor="none", s=12)
lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
ax.plot(lims, lims, "r--", label="Perfect prediction")
ax.set_title("Predicted vs Actual")
ax.set_xlabel("Actual value")
ax.set_ylabel("Predicted value")
ax.legend()
ax.grid(True, alpha=0.3)

# PANEL 2: Residuals (errors). Should scatter randomly around 0.
ax = axes[1]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, alpha=0.3, edgecolor="none", s=12)
ax.axhline(0, color="red", linestyle="--")
ax.set_title("Residuals vs Predicted")
ax.set_xlabel("Predicted value")
ax.set_ylabel("Residual (actual - predicted)")
ax.grid(True, alpha=0.3)

# PANEL 3: Feature importances.
ax = axes[2]
imp_sorted = importance_table.sort_values("importance")
ax.barh(imp_sorted["feature"], imp_sorted["importance"], color="steelblue", edgecolor="black")
ax.set_title("Feature Importances")
ax.set_xlabel("Importance")
ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("random_forest_regression_results.png", dpi=140, bbox_inches="tight")
print("Saved plot to 'random_forest_regression_results.png'.")

plt.show()


if __name__ == "__main__":
    print("\nDone. Scroll up and read each section slowly, like a guided tutorial.")
