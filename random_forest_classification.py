"""
RANDOM FOREST FOR CLASSIFICATION — A LEARNING WALKTHROUGH
=========================================================

THE BIG IDEA
------------
A single Decision Tree learns a series of yes/no questions to split the data
into classes. It is easy to understand, but a single deep tree tends to
memorize the training data (overfitting) and performs poorly on new data.

A Random Forest fixes this by training MANY decision trees and combining them:

    prediction = majority vote of all the trees

Like Logistic Regression, a Random Forest predicts a CLASS, such as:

    0 / 1
    No / Yes
    Not Fraud / Fraud
    Benign / Malignant

But it thinks in a completely different way.

HOW RANDOM FOREST THINKS
------------------------
Step 1: Build many decision trees (for example, 300 trees).

Step 2: Make each tree slightly different so they do not all agree blindly.
        Two sources of randomness are used:

        a) Bagging (Bootstrap Aggregating):
           Each tree is trained on a random sample of the rows,
           drawn WITH replacement. So each tree sees a slightly
           different version of the data.

        b) Random feature selection:
           At each split, a tree is only allowed to consider a random
           subset of the features, not all of them.

Step 3: To predict a class, ask every tree to vote, then take the majority.
        The probability of a class is simply the fraction of trees that
        voted for it.

Because the trees make different mistakes, their errors tend to cancel out
when averaged. This is why a forest is usually far more accurate and stable
than a single tree.

RANDOM FOREST vs LOGISTIC REGRESSION
------------------------------------
- Logistic Regression learns ONE intercept and a coefficient per feature,
  then uses a sigmoid to produce a probability.
- Random Forest learns NO coefficients. It learns split rules across many
  trees and votes.
- Logistic Regression assumes a roughly linear relationship.
  Random Forest can capture non-linear patterns and feature interactions
  automatically.
- Logistic Regression usually benefits from feature scaling.
  Random Forest does NOT need feature scaling, because splits only care
  about the order of values, not their magnitude.

WHAT YOU WILL LEARN IN THIS FILE
--------------------------------
1. What classification data looks like.
2. Why Random Forest does not require feature scaling.
3. How to train a Random Forest model using scikit-learn.
4. How to read feature importances (the Random Forest equivalent of
   "which features matter most").
5. How prediction works as a vote across many trees.
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

    python random_forest_classification.py

The script will also save a plot file:

    random_forest_classification_results.png
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

# RandomForestClassifier is the classification algorithm we are learning.
from sklearn.ensemble import RandomForestClassifier

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
print("- Train set: used by the model to grow its decision trees.")
print("- Test set : unseen data used to check whether the model generalizes.")


# -----------------------------------------------------------------------------
# 5. CREATE AND TRAIN THE RANDOM FOREST MODEL
# -----------------------------------------------------------------------------

print_section("3. CREATE AND TRAIN THE RANDOM FOREST MODEL")

# Unlike Logistic Regression, Random Forest does NOT need feature scaling.
# A tree split only compares "is this feature value above or below a threshold",
# so the units and magnitude of a feature do not matter.
# That is why we do not wrap this model in a StandardScaler pipeline.
#
# Key hyperparameters explained:
# - n_estimators : how many trees to grow. More trees = more stable, but slower.
# - max_depth    : how deep each tree can grow. None means grow until pure.
# - max_features : how many features each split may consider (adds randomness).
# - random_state : fixes the randomness so results are reproducible.
# - n_jobs=-1    : use all CPU cores to train trees in parallel.

model = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,
    max_features="sqrt",
    random_state=42,
    n_jobs=-1,
)

# .fit() is where training happens.
# The forest grows many decision trees, each on a random bootstrap sample.
model.fit(X_train, y_train)

print("Model training completed.")
print("Model configuration:")
print(model)
print(f"\nNumber of trees in the forest: {len(model.estimators_)}")


# -----------------------------------------------------------------------------
# 6. INSPECT FEATURE IMPORTANCES
# -----------------------------------------------------------------------------

print_section("4. WHAT THE MODEL LEARNED: FEATURE IMPORTANCES")

# Random Forest has no intercept and no coefficients.
# Instead it reports "feature importances": how much each feature helped the
# trees reduce impurity (make cleaner splits) across the whole forest.
#
# Important differences from Logistic Regression coefficients:
# - Importances are always >= 0 and sum to 1.0.
# - They have NO sign, so they do not tell you the direction (class 0 vs class 1),
#   only the strength of influence.
importances = model.feature_importances_

print("Feature importance meaning:")
print("- Each importance is how much a feature contributed to clean splits.")
print("- All importances are non-negative and add up to 1.0.")
print("- Higher value means the feature was more useful to the forest.")
print("- Unlike coefficients, importances do NOT indicate direction.")

importance_table = pd.DataFrame(
    {
        "feature": feature_names,
        "importance": importances,
    }
).sort_values("importance", ascending=False)

print(f"\nDo importances sum to 1.0? Total = {importances.sum():.4f}")

print("\nTop 10 features by importance:")
print(importance_table.head(10).to_string(index=False))

print("\nBeginner interpretation:")
print("- Bigger importance means the feature was used for more useful splits.")
print("- Importance says nothing about which class the feature points to.")
print("- Correlated features can share/split importance, so read this carefully.")


# -----------------------------------------------------------------------------
# 7. MAKE PREDICTIONS
# -----------------------------------------------------------------------------

print_section("5. MAKE PREDICTIONS ON TEST DATA")

# predict() returns final class labels: 0 or 1 (the majority vote of the trees).
y_pred = model.predict(X_test)

# predict_proba() returns probabilities for each class.
# For a Random Forest, the probability of class 1 is simply the fraction of
# trees that voted for class 1.
# Column 0 = probability of class 0.
# Column 1 = probability of class 1.
y_proba = model.predict_proba(X_test)
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
print("- probability_class_1 is the fraction of trees voting for class 1.")
print("- predict converts that probability into a class using threshold 0.50.")


# -----------------------------------------------------------------------------
# 8. INSPECT THE VOTE FOR ONE ROW
# -----------------------------------------------------------------------------

print_section("6. HOW THE VOTE WORKS: PROVE WHAT predict_proba() IS DOING")

# Choose the first test row.
one_sample = X_test[0:1]
one_sample_actual = y_test[0]

# Ask every individual tree in the forest for its own prediction on this row.
# Each entry in model.estimators_ is a single trained DecisionTreeClassifier.
individual_votes = np.array(
    [tree.predict(one_sample)[0] for tree in model.estimators_]
)

votes_for_class_0 = int(np.sum(individual_votes == 0))
votes_for_class_1 = int(np.sum(individual_votes == 1))
total_trees = len(model.estimators_)

# Manually compute the class-1 probability as the fraction of class-1 votes.
manual_probability_class_1 = votes_for_class_1 / total_trees
manual_predicted_class = 1 if manual_probability_class_1 >= 0.50 else 0

sklearn_probability_class_1 = model.predict_proba(one_sample)[0, 1]
sklearn_predicted_class = model.predict(one_sample)[0]

print(f"Actual class: {one_sample_actual} ({class_names[one_sample_actual]})")
print(f"Total trees in the forest: {total_trees}")
print(f"Trees voting class 0: {votes_for_class_0}")
print(f"Trees voting class 1: {votes_for_class_1}")
print(f"Manual probability for class 1 (votes_1 / total): {manual_probability_class_1:.4f}")
print(f"Manual predicted class: {manual_predicted_class} ({class_names[manual_predicted_class]})")
print(f"sklearn probability for class 1: {sklearn_probability_class_1:.4f}")
print(f"sklearn predicted class: {sklearn_predicted_class} ({class_names[sklearn_predicted_class]})")

print("\nThis proves the core logic:")
print("1. Every tree votes for a class.")
print("2. The class probability is the fraction of trees voting for it.")
print("3. Apply a threshold (default 0.50) to choose the final class.")
print("Note: sklearn averages each tree's own probability estimates, so the")
print("numbers can differ very slightly from a pure hard-vote count.")


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

print("By default, the forest predicts class 1 when the vote fraction >= 0.50.")
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
fig.suptitle("Random Forest Classification Results", fontsize=14, fontweight="bold")

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

# PANEL 3: Top feature importances
ax = axes[2]
top_n = 10
top_importance = importance_table.head(top_n).sort_values("importance")
ax.barh(top_importance["feature"], top_importance["importance"], color="steelblue", edgecolor="black")
ax.set_title("Top 10 Feature Importances")
ax.set_xlabel("Importance")
ax.grid(True, axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig("random_forest_classification_results.png", dpi=140, bbox_inches="tight")
print("Saved plot to 'random_forest_classification_results.png'.")

# If you run this locally, plt.show() will open the chart window.
# In some notebook/server environments, it may display inline or do nothing.
plt.show()


# -----------------------------------------------------------------------------
# 12. MINI CHEAT SHEET
# -----------------------------------------------------------------------------

print_section("10. MINI CHEAT SHEET")

print("Random Forest idea:")
print("    Grow many decision trees on random samples of rows and features.")
print("    probability_class_1 = fraction of trees voting for class 1")
print("    predicted_class = 1 if probability >= threshold else 0")

print("\nCommon sklearn methods:")
print("    model.fit(X_train, y_train)        -> train the forest")
print("    model.predict(X_test)              -> predict class labels (majority vote)")
print("    model.predict_proba(X_test)        -> predict class probabilities (vote fraction)")
print("    model.feature_importances_         -> how useful each feature was")
print("    model.estimators_                  -> the list of individual trees")

print("\nKey hyperparameters:")
print("    n_estimators  -> number of trees (more = stable but slower)")
print("    max_depth     -> how deep each tree can grow")
print("    max_features  -> features considered per split (adds randomness)")
print("    n_jobs=-1     -> train trees in parallel across all CPU cores")

print("\nWhen Random Forest is a good choice:")
print("- You expect non-linear relationships or feature interactions.")
print("- You want a strong model without much tuning.")
print("- You do not want to bother with feature scaling.")
print("- You want built-in feature importance.")

print("\nThings to remember:")
print("1. Random Forest is an ensemble of many decision trees.")
print("2. It predicts by majority vote; probability is the vote fraction.")
print("3. It has feature importances instead of coefficients (no direction/sign).")
print("4. Feature scaling is NOT required for tree-based models.")
print("5. More trees usually help stability but cost more time and memory.")
print("6. Accuracy alone can be misleading if classes are imbalanced.")
print("7. Precision and recall are often more useful for business decisions.")
print("8. Always evaluate on unseen test data.")


if __name__ == "__main__":
    print("\nDone. Scroll up and read each section slowly, like a guided tutorial.")
