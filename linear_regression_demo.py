"""
Linear Regression Prediction — A Learning Example
==================================================

Linear regression finds the best straight line through your data:

        y = m * x + b

    y  ->  the value we want to PREDICT   (target / dependent variable)
    x  ->  the value we already KNOW      (feature / independent variable)
    m  ->  the slope        (how much y changes when x increases by 1)
    b  ->  the intercept    (value of y when x is 0)

The model "learns" the best m and b from example data, then uses them
to predict y for brand-new x values it has never seen.

Scenario in this demo: predict a person's SALARY from their YEARS OF EXPERIENCE.

Run:   python linear_regression_demo.py
Needs: pip install numpy scikit-learn matplotlib
"""

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ----------------------------------------------------------------------
# 1. THE DATA
# ----------------------------------------------------------------------
# X = the feature (years of experience). scikit-learn expects X to be
#     2D: one row per sample, one column per feature -> shape (n, 1).
# y = the target we want to predict (salary in thousands).
X = np.array([[1], [2], [3], [4], [5], [6], [7], [8], [9], [10]])
y = np.array([30, 35, 40, 52, 55, 62, 68, 74, 80, 88])

print("Feature X (years):", X.ravel())
print("Target  y (salary):", y)
print()


# ----------------------------------------------------------------------
# 2. SPLIT INTO TRAINING AND TEST SETS
# ----------------------------------------------------------------------
# We train the model on one part of the data, then test it on data it
# has never seen. This tells us whether it truly LEARNED the pattern
# rather than just memorizing.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)


# ----------------------------------------------------------------------
# 3. TRAIN (FIT) THE MODEL
# ----------------------------------------------------------------------
# .fit() runs the math that finds the best slope (m) and intercept (b)
# by minimizing the total squared error between the line and the points.
model = LinearRegression()
model.fit(X_train, y_train)

slope = model.coef_[0]        # m
intercept = model.intercept_  # b
print(f"Learned equation:  salary = {slope:.2f} * years + {intercept:.2f}")
print()


# ----------------------------------------------------------------------
# 4. PREDICT
# ----------------------------------------------------------------------
# Use the learned line to predict salaries for the test set...
y_pred = model.predict(X_test)

print("Predictions on unseen test data:")
for years, actual, predicted in zip(X_test.ravel(), y_test, y_pred):
    print(f"  {years} yrs -> actual: {actual}, predicted: {predicted:.1f}")
print()

# ...and predict a brand-new value the model has never seen.
new_experience = np.array([[12]])
future_salary = model.predict(new_experience)
print(f"Predicted salary for 12 years of experience: {future_salary[0]:.1f}k")
print()


# ----------------------------------------------------------------------
# 5. EVALUATE HOW GOOD THE MODEL IS
# ----------------------------------------------------------------------
# MSE  : average squared error (lower is better; 0 = perfect).
# R^2  : fraction of variation explained (1.0 = perfect, 0 = useless).
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R^2):          {r2:.3f}")
print()


# ----------------------------------------------------------------------
# 6. (OPTIONAL) VISUALIZE THE FITTED LINE
# ----------------------------------------------------------------------
# Comment this block out if you don't have matplotlib installed.
try:
    import matplotlib.pyplot as plt

    plt.scatter(X, y, color="blue", label="Actual data")
    plt.plot(X, model.predict(X), color="red", label="Regression line")
    plt.xlabel("Years of experience")
    plt.ylabel("Salary (thousands)")
    plt.title("Linear Regression: Salary vs. Experience")
    plt.legend()
    plt.tight_layout()
    plt.savefig("linear_regression_plot.png", dpi=100)
    print("Plot saved as linear_regression_plot.png")
except ImportError:
    print("matplotlib not installed — skipping the plot.")
