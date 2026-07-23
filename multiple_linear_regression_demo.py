"""
================================================================================
MULTIPLE LINEAR REGRESSION — A Learning Walkthrough
================================================================================

GOAL OF THIS FILE
-----------------
Teach you, line by line, what "multiple linear regression" is, and — most
importantly — how the COEFFICIENTS and the INTERCEPT are used to make
predictions.

THE BIG IDEA (read this first)
------------------------------
"Simple" linear regression uses ONE input (feature) to predict an output:

        y = m*x + b

"Multiple" linear regression uses SEVERAL inputs at once:

        y = b0 + b1*x1 + b2*x2 + b3*x3 + ... + bn*xn

Where:
    y   -> the value we want to predict          (the "target" / "dependent variable")
    x1..xn -> the input features                 (the "independent variables")
    b0  -> the INTERCEPT (a.k.a. bias)           -> y when ALL features are 0
    b1..bn -> the COEFFICIENTS (a.k.a. weights)  -> how much y changes when a
                                                    feature increases by 1 unit

So the model is just a weighted sum of your inputs, plus a constant offset.
Training the model = finding the best b0, b1, ..., bn so predictions are as
close as possible to the real y values.

We'll use a concrete example: predicting a house PRICE from three features:
    x1 = size in square feet
    x2 = number of bedrooms
    x3 = age of the house in years
================================================================================
"""

# ------------------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------------------
# numpy: handles the numeric arrays/matrices behind the scenes.
import numpy as np

# matplotlib: the standard plotting library in Python. `pyplot` is the module
# that gives us MATLAB-style commands like plt.scatter(), plt.plot(), etc.
import matplotlib.pyplot as plt

# scikit-learn (sklearn) is the standard ML library in Python.
# LinearRegression is the class that finds the coefficients + intercept for us.
from sklearn.linear_model import LinearRegression

# We'll split data into a training part and a testing part so we can check
# whether the model generalizes to data it has NOT seen.
from sklearn.model_selection import train_test_split

# Metrics to score how good the predictions are.
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------------------------------------------
# 2. BUILD SOME SAMPLE DATA
# ------------------------------------------------------------------------------
# In real life this data would come from a CSV/database. Here we invent it.
#
# X is the FEATURE MATRIX. Shape = (number_of_samples, number_of_features).
# Each ROW is one house. Each COLUMN is one feature (size, bedrooms, age).
#
# Think of it as a table:
#       size(ft^2)   bedrooms   age(yrs)
#       ----------   --------   --------
#          1400          3          20
#          1600          3          15
#           ...         ...        ...
X = np.array([
    [1400, 3, 20],
    [1600, 3, 15],
    [1700, 4, 18],
    [1875, 4, 12],
    [1100, 2, 30],
    [1550, 3, 22],
    [2350, 4,  5],
    [2450, 5,  3],
    [1425, 3, 25],
    [1700, 3, 10],
    [1250, 2, 28],
    [2000, 4,  8],
], dtype=float)

# y is the TARGET vector: the actual sale price (in thousands of dollars) for
# each corresponding row in X. Length must equal the number of rows in X.
y = np.array([
    245, 312, 279, 308, 199, 219, 405, 460, 234, 330, 205, 375
], dtype=float)

# Human-friendly names for each column, used later when we print coefficients.
feature_names = ["size_sqft", "bedrooms", "age_years"]


# ------------------------------------------------------------------------------
# 3. SPLIT INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------------------------
# We train on one portion and evaluate on another. This tells us whether the
# model learned a real pattern or just memorized the data.
#
# test_size=0.25 -> keep 25% of rows for testing, 75% for training.
# random_state=42 -> a fixed "seed" so the split is the same every run
#                    (reproducible results; the exact number doesn't matter).
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ------------------------------------------------------------------------------
# 4. CREATE AND TRAIN THE MODEL
# ------------------------------------------------------------------------------
# Create an (untrained) model object.
model = LinearRegression()

# .fit() is where the LEARNING happens. Under the hood sklearn solves for the
# intercept (b0) and coefficients (b1..bn) that minimize the total squared
# error between predictions and the real y_train values ("least squares").
#
# After this line, `model` knows its best-fit b0 and b1..bn.
model.fit(X_train, y_train)


# ------------------------------------------------------------------------------
# 5. INSPECT WHAT THE MODEL LEARNED  <-- THE HEART OF THIS LESSON
# ------------------------------------------------------------------------------
# model.intercept_  -> a single number: b0 (the intercept / bias).
# model.coef_       -> an array: [b1, b2, b3], one coefficient per feature,
#                      in the SAME ORDER as the columns of X.
intercept = model.intercept_
coefficients = model.coef_

print("=" * 70)
print("WHAT THE MODEL LEARNED")
print("=" * 70)
print(f"Intercept (b0): {intercept:.3f}")
print("Coefficients (one per feature):")
for name, coef in zip(feature_names, coefficients):
    # HOW TO READ EACH COEFFICIENT:
    # "Holding every other feature fixed, increasing THIS feature by 1 unit
    #  changes the predicted price by `coef` (in thousands of dollars)."
    #
    # Sign matters:
    #   positive coef -> feature pushes the prediction UP
    #   negative coef -> feature pushes the prediction DOWN
    # e.g. age_years will likely be negative: older house -> lower price.
    direction = "increases" if coef >= 0 else "decreases"
    print(f"   {name:>10}: {coef:+.3f}  "
          f"(each +1 unit {direction} price by {abs(coef):.3f})")

# So the learned prediction formula (the actual equation) is literally:
#
#   price = b0 + b1*size + b2*bedrooms + b3*age
#
# Let's print it with the real learned numbers:
equation = f"price = {intercept:.2f}"
for name, coef in zip(feature_names, coefficients):
    equation += f" + ({coef:.3f} * {name})"
print("\nLearned equation:")
print("   " + equation)


# ------------------------------------------------------------------------------
# 6. HOW A PREDICTION IS MADE — BY HAND vs. BY sklearn
# ------------------------------------------------------------------------------
# To PROVE that a prediction is nothing more than "intercept + weighted sum of
# features", let's compute one prediction manually and compare it to sklearn.
#
# Take one brand-new house we want to price:
new_house = np.array([[1650, 3, 14]])   # size=1650, bedrooms=3, age=14
                                        # (2D shape [[...]] because sklearn
                                        #  expects a 2D array: rows x features)

# --- Manual calculation ---
# Multiply each feature by its coefficient, sum them, then add the intercept.
# np.dot does the "b1*x1 + b2*x2 + b3*x3" multiply-and-add in one step.
manual_prediction = intercept + np.dot(new_house[0], coefficients)

# --- sklearn calculation ---
# .predict() applies the exact same math internally for every row you give it.
sklearn_prediction = model.predict(new_house)[0]

print("\n" + "=" * 70)
print("MAKING A PREDICTION FOR A NEW HOUSE")
print("=" * 70)
print(f"New house features: size={new_house[0][0]:.0f}, "
      f"bedrooms={new_house[0][1]:.0f}, age={new_house[0][2]:.0f}")
print(f"Manual  prediction: {manual_prediction:.3f} (thousand $)")
print(f"sklearn prediction: {sklearn_prediction:.3f} (thousand $)")
print("Notice both numbers match -> predict() is just the equation above.")


# ------------------------------------------------------------------------------
# 7. EVALUATE THE MODEL ON THE TEST SET
# ------------------------------------------------------------------------------
# Now use the held-out test data (rows the model never trained on) to see how
# well those learned coefficients generalize.
y_pred = model.predict(X_test)

# Root Mean Squared Error: average size of the prediction error, in the same
# units as y (thousands of $). Lower is better.
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# R^2 ("R-squared"): fraction of the variation in price explained by the model.
#   1.0  -> perfect predictions
#   0.0  -> no better than always guessing the average price
#   <0   -> worse than guessing the average
r2 = r2_score(y_test, y_pred)

print("\n" + "=" * 70)
print("MODEL PERFORMANCE ON UNSEEN TEST DATA")
print("=" * 70)
print(f"RMSE (avg error): {rmse:.3f} thousand $")
print(f"R^2  (fit quality, 1.0 is perfect): {r2:.3f}")

# Side-by-side comparison of predicted vs actual for the test houses.
print("\nPredicted vs Actual (test set):")
print(f"{'predicted':>12} {'actual':>12} {'error':>12}")
for pred, actual in zip(y_pred, y_test):
    print(f"{pred:>12.1f} {actual:>12.1f} {pred - actual:>+12.1f}")


# ------------------------------------------------------------------------------
# 7b. PLOT THE RESULTS
# ------------------------------------------------------------------------------
# A model with 3 features lives in 4-dimensional space (3 inputs + 1 output),
# which we can't draw directly. So instead of trying to plot the "line", we plot
# things that DO reveal how well the model works. We'll make three panels.
#
# plt.subplots(1, 3, ...) creates one figure ("fig") with a row of 3 plotting
# areas ("axes"). `axes` is an array; axes[0], axes[1], axes[2] are each a panel.
fig, axes = plt.subplots(1, 3, figsize=(16, 5))
fig.suptitle("Multiple Linear Regression — Results", fontsize=14, fontweight="bold")

# --- PANEL 1: Predicted vs Actual -------------------------------------------
# The single best diagnostic plot for any regression.
# Each dot is one test house: x = the TRUE price, y = the model's PREDICTION.
# If the model were perfect, every dot would sit exactly on the diagonal line
# y = x. The closer the dots hug that line, the better the model.
ax = axes[0]
ax.scatter(y_test, y_pred, color="steelblue", s=70, edgecolor="black", zorder=3)

# Draw the "perfect prediction" reference line (y = x) across the data range.
lo = min(y_test.min(), y_pred.min())
hi = max(y_test.max(), y_pred.max())
ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="perfect prediction (y = x)")

ax.set_xlabel("Actual price (thousand $)")
ax.set_ylabel("Predicted price (thousand $)")
ax.set_title("Predicted vs Actual")
ax.legend()
ax.grid(True, alpha=0.3)

# --- PANEL 2: Residuals (the errors) ----------------------------------------
# A "residual" = actual - predicted = how far off each prediction was.
# We plot residuals against the predicted value. A GOOD model shows residuals
# scattered randomly around the horizontal zero line (no pattern). A visible
# curve or funnel shape would hint the straight-line model is missing something.
ax = axes[1]
residuals = y_test - y_pred
ax.scatter(y_pred, residuals, color="darkorange", s=70, edgecolor="black", zorder=3)

# The zero line = "no error". Points above it were under-predicted, below it
# were over-predicted.
ax.axhline(y=0, color="red", linestyle="--", linewidth=2, label="zero error")

ax.set_xlabel("Predicted price (thousand $)")
ax.set_ylabel("Residual (actual - predicted)")
ax.set_title("Residual Plot")
ax.legend()
ax.grid(True, alpha=0.3)

# --- PANEL 3: The learned coefficients --------------------------------------
# This directly visualizes THE HEART OF THE LESSON: each feature's weight.
# Taller bar = bigger influence on the prediction. Bars above zero push price
# up; bars below zero push it down (e.g. age is expected to be negative).
# (Reminder: bar heights are only fairly comparable if features share a scale —
#  see the caution note in the takeaways below.)
ax = axes[2]
colors = ["seagreen" if c >= 0 else "indianred" for c in coefficients]
ax.bar(feature_names, coefficients, color=colors, edgecolor="black")
ax.axhline(y=0, color="black", linewidth=0.8)
ax.set_ylabel("Coefficient value")
ax.set_title("Learned Coefficients (feature weights)")
ax.grid(True, axis="y", alpha=0.3)

# tight_layout() tidies spacing so titles/labels don't overlap.
plt.tight_layout()

# Save the figure to a PNG file so you can open it even without a screen popup.
plt.savefig("regression_results.png", dpi=120)
print("\nSaved plot to 'regression_results.png'.")

# plt.show() opens an interactive window with the plots. If you're running in an
# environment without a display, the saved PNG above is your result.
plt.show()


# ------------------------------------------------------------------------------
# 8. KEY TAKEAWAYS (summary you can re-read anytime)
# ------------------------------------------------------------------------------
# 1) A multiple linear regression model IS just this equation:
#        y = intercept + (coef1 * feature1) + (coef2 * feature2) + ...
#
# 2) TRAINING (.fit) = finding the intercept + coefficients that make the
#    predictions fit the training data as closely as possible.
#
# 3) The INTERCEPT (model.intercept_) is the predicted value when every
#    feature equals 0 — the baseline/starting point.
#
# 4) Each COEFFICIENT (model.coef_) is the "weight" of one feature: how much
#    the prediction moves per +1 unit of that feature, holding others fixed.
#    Sign tells you direction (up/down); magnitude tells you strength.
#    (CAUTION: to compare coefficient magnitudes fairly, features should be on
#     similar scales — otherwise a feature measured in thousands will look more
#     "important" than one measured in single digits just due to units.)
#
# 5) PREDICTING (.predict) = plugging new feature values into that same
#    equation. Nothing more magical than a weighted sum plus a constant.
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    # Everything above already runs top-to-bottom, so running this file
    # directly (python multiple_linear_regression_demo.py) prints the full
    # walkthrough. This guard just makes the intent explicit.
    print("\nDone. Scroll up to follow the walkthrough from top to bottom.")
