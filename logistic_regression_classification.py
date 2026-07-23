"""
LOGISTIC REGRESSION FOR CLASSIFICATION — A LEARNING WALKTHROUGH
================================================================

THE BIG IDEA
------------
Linear Regression predicts a continuous number, such as house price.
Logistic Regression predicts a CLASS, such as:

    0 / 1
    No / Yes
    Not Fraud / Fraud
    Customer will not churn / Customer will churn
    Benign / Malignant

Even though the name contains "regression", Logistic Regression is mainly used
for classification.

HOW LOGISTIC REGRESSION THINKS
------------------------------
Step 1: Create a linear score from the input features:

    z = intercept + coef1*x1 + coef2*x2 + ... + coefN*xN

Step 2: Convert that score into a probability using the sigmoid function:

    probability = 1 / (1 + exp(-z))

Step 3: Convert probability into a class using a threshold:

    if probability >= 0.50 -> predict class 1
    else                   -> predict class 0

So the model is still learning an intercept and coefficients, but the final
output is a probability/class instead of a continuous value.

WHAT YOU WILL LEARN IN THIS FILE
--------------------------------
1. What classification data looks like.
2. Why feature scaling is commonly used with Logistic Regression.
3. How to train a Logistic Regression model using scikit-learn.
4. How to read intercept and coefficients.
5. How prediction works manually using sigmoid.
6. How to evaluate classification models using:
   - Accuracy
   - Precision
   - Recall
   - F1-score
   - Confusion matrix
   - ROC-AUC
7. How changing the probability threshold changes predictions.
8. How to visualize results.

DATASET USED
------------
We use scikit-learn's built-in Breast Cancer dataset.
This is a standard binary classification dataset.
The target classes are:

    0 = malignant
    1 = benign

NOTE: This is for learning only. Do not use this demo as a medical diagnostic
system.

HOW TO RUN
----------
Install dependencies if needed:

    pip install numpy pandas matplotlib scikit-learn

Run:

    python logistic_regression_classification.py

The script will also save a plot file:

    logistic_regression_classification_results.png
"""

# -----------------------------------------------------------------------------
# 1. IMPORTS
# -----------------------------------------------------------------------------

# numpy handles numerical arrays and math operations.
import numpy as np

# pandas is useful for tabular display and inspection.
import pandas as pd

# matplotlib is used to create plots.
import matplotlib.pyplot as plt

# scikit-learn has built-in datasets and machine learning algorithms.
from sklearn.datasets import load_breast_cancer

# train_test_split splits data into training and testing portions.
from sklearn.model_selection import train_test_split

# StandardScaler scales each feature to mean 0 and standard deviation 1.
# Logistic Regression often trains better when feature scales are comparable.
from sklearn.preprocessing import StandardScaler

# Pipeline chains preprocessing and model training safely.
# This avoids data leakage because the scaler is fitted only on training data.
from sklearn.pipeline import Pipeline

# LogisticRegression is the classification algorithm we are learning.
from sklearn.linear_model import LogisticRegression

# Classification evaluation metrics.
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)


# -----------------------------------------------------------------------------
# 2. HELPER FUNCTIONS
# -----------------------------------------------------------------------------

def sigmoid(z):
    """
    Convert a raw linear score into a probability between 0 and 1.

    If z is very positive, sigmoid(z) is close to 1.
    If z is very negative, sigmoid(z) is close to 0.
    If z is 0, sigmoid(z) is exactly 0.5.
    """
    return 1 / (1 + np.exp(-z))


def print_section(title):
    """Print a section heading so the console output is easy to read."""
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def explain_confusion_matrix(cm, class_names):
    """
    Print a beginner-friendly explanation of a binary confusion matrix.

    For sklearn confusion_matrix with labels ordered as [0, 1]:

        [[TN, FP],
         [FN, TP]]

    Here, class 1 is treated as the positive class by default for metrics such
    as precision, recall, and F1-score.
    """
    tn, fp, fn, tp = cm.ravel()
    print("Confusion Matrix layout for binary classification:")
    print("                 Predicted Class 0    Predicted Class 1")
    print(f"Actual Class 0        TN={tn:<5}              FP={fp:<5}")
    print(f"Actual Class 1        FN={fn:<5}              TP={tp:<5}")
    print()
    print(f"Class 0 = {class_names[0]}")
    print(f"Class 1 = {class_names[1]}")
    print()
    print("How to read it:")
    print("- TN: Actual class 0, predicted class 0")
    print("- FP: Actual class 0, predicted class 1")
    print("- FN: Actual class 1, predicted class 0")
    print("- TP: Actual class 1, predicted class 1")


# -----------------------------------------------------------------------------
# 3. LOAD THE DATASET
# -----------------------------------------------------------------------------

print_section("1. LOAD DATA")

cancer = load_breast_cancer()

# X is the feature matrix.
# Each row is one sample.
# Each column is one numeric measurement.
X = cancer.data

# y is the target vector.
# Each value is the known class label for the corresponding row in X.
y = cancer.target

feature_names = cancer.feature_names
class_names = cancer.target_names  # array(['malignant', 'benign'])

print(f"Number of samples/rows: {X.shape[0]}")
print(f"Number of features/columns: {X.shape[1]}")
print(f"Class names: 0 = {class_names[0]}, 1 = {class_names[1]}")

# Convert first few rows to a DataFrame for easier reading.
df_preview = pd.DataFrame(X, columns=feature_names)
df_preview["target"] = y
df_preview["target_name"] = df_preview["target"].map({0: class_names[0], 1: class_names[1]})

print("\nFirst 5 rows of the dataset:")
print(df_preview.head())

print("\nClass distribution:")
print(df_preview["target_name"].value_counts())


# -----------------------------------------------------------------------------
# 4. TRAIN/TEST SPLIT
# -----------------------------------------------------------------------------

print_section("2. SPLIT DATA INTO TRAINING AND TESTING SETS")

# We use stratify=y so the train and test sets keep a similar class balance.
# This is important in classification.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    random_state=42,
    stratify=y,
)

print(f"Training rows: {X_train.shape[0]}")
print(f"Testing rows : {X_test.shape[0]}")

print("\nWhy split the data?")
print("- Train set: used by the model to learn intercept and coefficients.")
print("- Test set : unseen data used to check whether the model generalizes.")


# -----------------------------------------------------------------------------
# 5. CREATE A SAFE ML PIPELINE
# -----------------------------------------------------------------------------

print_section("3. CREATE AND TRAIN THE LOGISTIC REGRESSION MODEL")

# Logistic Regression uses optimization internally.
# Scaling helps because features can be measured in very different units.
# Example: radius, area, texture, etc. can have very different ranges.

model_pipeline = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        (
            "logistic_regression",
            LogisticRegression(
                max_iter=5000,
                random_state=42,
                solver="lbfgs",
            ),
        ),
    ]
)

# .fit() is where training happens.
# The model learns coefficients and intercept that best separate the classes.
model_pipeline.fit(X_train, y_train)

print("Model training completed.")
print("Pipeline steps:")
print(model_pipeline)


# -----------------------------------------------------------------------------
# 6. INSPECT INTERCEPT AND COEFFICIENTS
# -----------------------------------------------------------------------------

print_section("4. WHAT THE MODEL LEARNED: INTERCEPT AND COEFFICIENTS")

# Get the trained logistic regression object from the pipeline.
log_reg_model = model_pipeline.named_steps["logistic_regression"]

intercept = log_reg_model.intercept_[0]
coefficients = log_reg_model.coef_[0]

print(f"Intercept: {intercept:+.4f}")
print("\nCoefficient meaning:")
print("Each coefficient is attached to one feature.")
print("A positive coefficient pushes the model toward class 1.")
print("A negative coefficient pushes the model toward class 0.")
print("Important: Because we used StandardScaler, coefficient values refer to scaled features.")

coef_table = pd.DataFrame(
    {
        "feature": feature_names,
        "coefficient": coefficients,
        "absolute_strength": np.abs(coefficients),
    }
).sort_values("absolute_strength", ascending=False)

print("\nTop 10 features by absolute coefficient strength:")
print(coef_table.head(10).to_string(index=False))

print("\nBeginner interpretation:")
print("- Bigger absolute coefficient means the feature has stronger influence in this fitted model.")
print("- Sign tells direction: positive -> class 1, negative -> class 0.")
print("- Do not blindly treat this as universal feature importance; it is model- and data-dependent.")


# -----------------------------------------------------------------------------
# 7. MAKE PREDICTIONS
# -----------------------------------------------------------------------------

print_section("5. MAKE PREDICTIONS ON TEST DATA")

# predict() returns final class labels: 0 or 1.
y_pred = model_pipeline.predict(X_test)

# predict_proba() returns probabilities for each class.
# Column 0 = probability of class 0.
# Column 1 = probability of class 1.
y_proba = model_pipeline.predict_proba(X_test)
y_proba_class_1 = y_proba[:, 1]

prediction_table = pd.DataFrame(
    {
        "actual_class": y_test,
        "actual_name": [class_names[i] for i in y_test],
        "predicted_class": y_pred,
        "predicted_name": [class_names[i] for i in y_pred],
        "probability_class_1": y_proba_class_1,
    }
)

print("First 10 predictions:")
print(prediction_table.head(10).to_string(index=False))

print("\nNotice:")
print("- predict_proba gives probability.")
print("- predict converts probability into class using default threshold 0.50.")


# -----------------------------------------------------------------------------
# 8. MANUAL PREDICTION FOR ONE ROW
# -----------------------------------------------------------------------------

print_section("6. MANUAL PREDICTION: PROVE WHAT predict_proba() IS DOING")

# Choose the first test row.
one_sample_original = X_test[0:1]
one_sample_actual = y_test[0]

# To manually reproduce the model, we must apply the same scaler first.
scaler = model_pipeline.named_steps["scaler"]
one_sample_scaled = scaler.transform(one_sample_original)

# Raw score, also called logit:
# z = intercept + coefficient_1*x_1 + ... + coefficient_n*x_n
manual_z = intercept + np.dot(one_sample_scaled[0], coefficients)
manual_probability_class_1 = sigmoid(manual_z)
manual_predicted_class = 1 if manual_probability_class_1 >= 0.50 else 0

sklearn_probability_class_1 = model_pipeline.predict_proba(one_sample_original)[0, 1]
sklearn_predicted_class = model_pipeline.predict(one_sample_original)[0]

print(f"Actual class: {one_sample_actual} ({class_names[one_sample_actual]})")
print(f"Manual raw score z: {manual_z:+.4f}")
print(f"Manual probability for class 1: {manual_probability_class_1:.4f}")
print(f"Manual predicted class: {manual_predicted_class} ({class_names[manual_predicted_class]})")
print(f"sklearn probability for class 1: {sklearn_probability_class_1:.4f}")
print(f"sklearn predicted class: {sklearn_predicted_class} ({class_names[sklearn_predicted_class]})")

print("\nThis proves the core logic:")
print("1. Scale features")
print("2. Compute z = intercept + weighted sum")
print("3. Convert z to probability using sigmoid")
print("4. Apply threshold to choose class")


# -----------------------------------------------------------------------------
# 9. EVALUATE THE MODEL
# -----------------------------------------------------------------------------

print_section("7. MODEL PERFORMANCE METRICS")

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_proba_class_1)
cm = confusion_matrix(y_test, y_pred)

print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1-score : {f1:.4f}")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nMetric meaning:")
print("- Accuracy : Overall percentage of correct predictions.")
print("- Precision: When model predicts class 1, how often is it correct?")
print("- Recall   : Of all actual class 1 cases, how many did model catch?")
print("- F1-score : Balanced score combining precision and recall.")
print("- ROC-AUC  : Measures ranking quality across many thresholds.")

print("\nConfusion Matrix:")
print(cm)
print()
explain_confusion_matrix(cm, class_names)

print("\nFull classification report:")
print(classification_report(y_test, y_pred, target_names=class_names))


# -----------------------------------------------------------------------------
# 10. THRESHOLD TUNING
# -----------------------------------------------------------------------------

print_section("8. PROBABILITY THRESHOLD: 0.50 IS NOT ALWAYS THE ONLY CHOICE")

print("By default, Logistic Regression predicts class 1 when probability >= 0.50.")
print("But in real business problems, you may change the threshold depending on risk.")
print("Example:")
print("- Fraud use case: you may lower threshold to catch more fraud, increasing recall.")
print("- Spam use case : you may raise threshold to avoid wrongly blocking good emails.")

thresholds_to_try = [0.30, 0.50, 0.70]
threshold_rows = []

for threshold in thresholds_to_try:
    y_pred_threshold = (y_proba_class_1 >= threshold).astype(int)
    threshold_rows.append(
        {
            "threshold": threshold,
            "accuracy": accuracy_score(y_test, y_pred_threshold),
            "precision": precision_score(y_test, y_pred_threshold, zero_division=0),
            "recall": recall_score(y_test, y_pred_threshold, zero_division=0),
            "f1": f1_score(y_test, y_pred_threshold, zero_division=0),
        }
    )

threshold_table = pd.DataFrame(threshold_rows)
print("\nThreshold comparison:")
print(threshold_table.to_string(index=False))

print("\nKey takeaway:")
print("Changing threshold changes the precision/recall trade-off.")
print("There is no universally best threshold; it depends on business cost of false positives vs false negatives.")


# -----------------------------------------------------------------------------
# 11. VISUALIZE RESULTS
# -----------------------------------------------------------------------------

print_section("9. CREATE VISUALIZATIONS")

# ROC curve data.
fpr, tpr, roc_thresholds = roc_curve(y_test, y_proba_class_1)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Logistic Regression Classification Results", fontsize=14, fontweight="bold")

# PANEL 1: Confusion matrix heatmap
ax = axes[0]
im = ax.imshow(cm, cmap="Blues")
ax.set_title("Confusion Matrix")
ax.set_xlabel("Predicted class")
ax.set_ylabel("Actual class")
ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(class_names)
ax.set_yticklabels(class_names)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(j, i, cm[i, j], ha="center", va="center", color="black", fontsize=12)

fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# PANEL 2: ROC curve
ax = axes[1]
ax.plot(fpr, tpr, label=f"ROC curve (AUC = {roc_auc:.3f})", linewidth=2)
ax.plot([0, 1], [0, 1], "r--", label="Random guessing")
ax.set_title("ROC Curve")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate / Recall")
ax.legend()
ax.grid(True, alpha=0.3)

# PANEL 3: Top coefficients
ax = axes[2]
top_n = 10
top_coef = coef_table.head(top_n).sort_values("coefficient")
colors = ["seagreen" if c >= 0 else "indianred" for c in top_coef["coefficient"]]
ax.barh(top_coef["feature"], top_coef["coefficient"], color=colors, edgecolor="black")
ax.axvline(0, color="black", linewidth=0.8)
ax.set_title("Top 10 Coefficients")
ax.set_xlabel("Coefficient value")
ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("logistic_regression_classification_results.png", dpi=140, bbox_inches="tight")
print("Saved plot to 'logistic_regression_classification_results.png'.")

# If you run this locally, plt.show() will open the chart window.
# In some notebook/server environments, it may display inline or do nothing.
plt.show()


# -----------------------------------------------------------------------------
# 12. MINI CHEAT SHEET
# -----------------------------------------------------------------------------

print_section("10. MINI CHEAT SHEET")

print("Logistic Regression formula:")
print("    z = intercept + coef1*x1 + coef2*x2 + ... + coefN*xN")
print("    probability = 1 / (1 + exp(-z))")
print("    predicted_class = 1 if probability >= threshold else 0")

print("\nCommon sklearn methods:")
print("    model.fit(X_train, y_train)        -> train model")
print("    model.predict(X_test)              -> predict class labels")
print("    model.predict_proba(X_test)        -> predict class probabilities")
print("    model.named_steps[...]             -> access pipeline steps")

print("\nWhen Logistic Regression is a good baseline:")
print("- Binary classification problem")
print("- Need interpretable coefficients")
print("- Want probability output")
print("- Need a simple, fast baseline model")

print("\nThings to remember:")
print("1. Logistic Regression is classification, not continuous value prediction.")
print("2. It learns intercept and coefficients just like linear models.")
print("3. Sigmoid converts the linear score into probability.")
print("4. Threshold converts probability into class.")
print("5. Feature scaling is usually helpful.")
print("6. Accuracy alone can be misleading if classes are imbalanced.")
print("7. Precision and recall are often more useful for business decisions.")
print("8. Always evaluate on unseen test data.")


if __name__ == "__main__":
    print("\nDone. Scroll up and read each section slowly, like a guided tutorial.")
