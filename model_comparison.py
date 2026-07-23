"""
================================================================================
COMPARING ML MODELS ON THE SAME PROBLEM — A Learning Walkthrough
================================================================================

GOAL OF THIS FILE
-----------------
In `multiple_linear_regression_demo.py` you fit ONE model (linear regression)
to predict house prices. Here we ask the natural follow-up question:

        "Is a fancier model actually better for this data?"

We train THREE different regression models on the SAME house-price data and
score them side by side, so you can compare them fairly:

    1. Linear Regression   -> the simple weighted-sum baseline you already know.
    2. Random Forest        -> many decision trees averaged together (bagging).
    3. Gradient Boosting    -> trees built one after another, each fixing the
                               previous one's mistakes (boosting).

BIG IDEA
--------
There is no single "best" ML model. The right choice depends on the data. The
professional workflow is ALWAYS:
    (a) fit a simple baseline first (linear regression),
    (b) try a couple of stronger models,
    (c) compare them on data none of them trained on,
    (d) pick the simplest model that is "good enough".

WHY THESE THREE?
----------------
- Linear Regression assumes a straight-line relationship. Fast, interpretable,
  but can't capture curves or feature interactions on its own.
- Random Forest & Gradient Boosting are TREE ENSEMBLES. They split the data
  into if/else regions, so they can model nonlinear patterns and interactions
  automatically. On tabular data (rows & columns like ours) they are usually
  the top performers.

IMPORTANT CAVEAT ABOUT OUR TINY DATASET
---------------------------------------
We only have 12 houses (9 train / 3 test). That is FAR too small to draw real
conclusions — powerful models like forests can easily "overfit" (memorize) tiny
data and then score poorly on the 3 test rows. So treat the numbers below as a
DEMONSTRATION OF THE WORKFLOW, not as proof that one model wins. With real data
(thousands of rows) the tree ensembles typically pull clearly ahead.
================================================================================
"""

# ------------------------------------------------------------------------------
# 1. IMPORTS
# ------------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt

# The three model classes we will compare. All follow the SAME sklearn API:
#   model.fit(X, y)  -> learn from data
#   model.predict(X) -> make predictions
# That shared interface is exactly what makes comparing models so easy.
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


# ------------------------------------------------------------------------------
# 2. THE SAME SAMPLE DATA AS THE LINEAR REGRESSION DEMO
# ------------------------------------------------------------------------------
# Each ROW is one house. Columns = [size_sqft, bedrooms, age_years].
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

# Target: sale price in thousands of dollars.
y = np.array([
    245, 312, 279, 308, 199, 219, 405, 460, 234, 330, 205, 375
], dtype=float)

feature_names = ["size_sqft", "bedrooms", "age_years"]


# ------------------------------------------------------------------------------
# 3. SPLIT INTO TRAINING AND TESTING SETS
# ------------------------------------------------------------------------------
# Same split as the other file (random_state=42) so the comparison is fair:
# every model trains on the exact same rows and is tested on the exact same rows.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42
)


# ------------------------------------------------------------------------------
# 4. DEFINE THE MODELS TO COMPARE
# ------------------------------------------------------------------------------
# We store each model in a dictionary: name -> (untrained) model object.
# Because every sklearn model shares the same .fit()/.predict() interface, we can
# loop over them and treat them identically. This is the key trick for comparing.
#
# A quick note on the settings passed in:
#   n_estimators = how many trees to build (more = more capacity, slower).
#   random_state = fixes the internal randomness so results are reproducible.
models = {
    "Linear Regression": LinearRegression(),
    "Random Forest":     RandomForestRegressor(n_estimators=200, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=200, random_state=42),
}


# ------------------------------------------------------------------------------
# 5. TRAIN AND EVALUATE EACH MODEL IN A LOOP
# ------------------------------------------------------------------------------
# For each model we:
#   1. fit it on the training data,
#   2. predict on the held-out test data,
#   3. measure RMSE (avg error, lower is better) and R^2 (fit quality, 1 = perfect),
#   4. stash the results so we can plot and rank them afterwards.
results = {}   # name -> dict of metrics + predictions

print("=" * 70)
print("TRAINING AND SCORING EACH MODEL ON THE SAME DATA")
print("=" * 70)

for name, model in models.items():
    # --- train ---
    model.fit(X_train, y_train)

    # --- predict on unseen test rows ---
    y_pred = model.predict(X_test)

    # --- score ---
    # RMSE: Root Mean Squared Error, in the same units as price (thousand $).
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    # R^2: fraction of price variation explained. 1.0 perfect, 0 = guess average.
    r2 = r2_score(y_test, y_pred)

    results[name] = {"model": model, "y_pred": y_pred, "rmse": rmse, "r2": r2}

    print(f"\n{name}")
    print(f"   RMSE (avg error): {rmse:7.3f} thousand $   (lower is better)")
    print(f"   R^2  (fit, 1=perfect): {r2:7.3f}          (higher is better)")


# ------------------------------------------------------------------------------
# 6. RANK THE MODELS
# ------------------------------------------------------------------------------
# Sort by RMSE ascending -> the smallest average error is "best" here.
ranked = sorted(results.items(), key=lambda kv: kv[1]["rmse"])

print("\n" + "=" * 70)
print("LEADERBOARD (best = lowest RMSE on the test set)")
print("=" * 70)
print(f"{'rank':>4}  {'model':<20} {'RMSE':>10} {'R^2':>8}")
for i, (name, r) in enumerate(ranked, start=1):
    print(f"{i:>4}  {name:<20} {r['rmse']:>10.3f} {r['r2']:>8.3f}")

best_name = ranked[0][0]
print(f"\nBest on THIS tiny test set: {best_name}")
print("(Remember: with only 3 test rows this ranking is not statistically")
print(" meaningful — it just demonstrates the comparison workflow.)")


# ------------------------------------------------------------------------------
# 7. PLOT THE COMPARISON
# ------------------------------------------------------------------------------
# Two panels:
#   Panel 1: a bar chart of RMSE per model (quick visual leaderboard).
#   Panel 2: Predicted-vs-Actual for all models overlaid, so you can see WHERE
#            each model does well or poorly relative to the perfect y=x line.
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Model Comparison — House Price Prediction", fontsize=14, fontweight="bold")

# --- PANEL 1: RMSE bar chart -------------------------------------------------
ax = axes[0]
names = list(results.keys())
rmses = [results[n]["rmse"] for n in names]
# Highlight the winner (lowest bar) in green, the rest in gray.
bar_colors = ["seagreen" if n == best_name else "steelblue" for n in names]
bars = ax.bar(names, rmses, color=bar_colors, edgecolor="black")
ax.set_ylabel("RMSE (thousand $) — lower is better")
ax.set_title("Average Test Error by Model")
ax.grid(True, axis="y", alpha=0.3)
# Print the exact RMSE on top of each bar for readability.
for bar, val in zip(bars, rmses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
            f"{val:.1f}", ha="center", va="bottom", fontweight="bold")
ax.tick_params(axis="x", rotation=15)

# --- PANEL 2: Predicted vs Actual, all models overlaid -----------------------
ax = axes[1]
markers = {"Linear Regression": "o", "Random Forest": "s", "Gradient Boosting": "^"}
for name in names:
    ax.scatter(y_test, results[name]["y_pred"],
               s=90, edgecolor="black", alpha=0.8,
               marker=markers.get(name, "o"), label=name, zorder=3)

# The perfect-prediction reference line y = x.
all_vals = np.concatenate([y_test] + [results[n]["y_pred"] for n in names])
lo, hi = all_vals.min(), all_vals.max()
ax.plot([lo, hi], [lo, hi], "r--", linewidth=2, label="perfect prediction (y = x)")

ax.set_xlabel("Actual price (thousand $)")
ax.set_ylabel("Predicted price (thousand $)")
ax.set_title("Predicted vs Actual (all models)")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("model_comparison.png", dpi=120)
print("\nSaved plot to 'model_comparison.png'.")
plt.show()


# ------------------------------------------------------------------------------
# 8. KEY TAKEAWAYS
# ------------------------------------------------------------------------------
# 1) EVERY sklearn model shares the same .fit()/.predict() interface, so swapping
#    or comparing models is as easy as changing one line — that's why we could
#    loop over all three identically.
#
# 2) ALWAYS start with a simple baseline (Linear Regression). You can only claim
#    a complex model is "better" by comparing it against that baseline.
#
# 3) COMPARE ON HELD-OUT TEST DATA, never on the training data. A model that
#    memorizes training rows can look perfect there and still fail on new data.
#
# 4) TREE ENSEMBLES (Random Forest, Gradient Boosting) usually win on real
#    tabular data because they capture nonlinear patterns and feature
#    interactions automatically — but they need enough data to shine.
#
# 5) MORE COMPLEX IS NOT ALWAYS BETTER. On tiny datasets (like our 12 houses),
#    simple models can match or beat powerful ones, and they are easier to
#    explain. Pick the simplest model that is good enough.
#
# NEXT STEPS you could try:
#   - Add more data rows and watch the ranking shift.
#   - Add XGBoost / LightGBM (pip install xgboost lightgbm) to the `models` dict.
#   - Use cross-validation (sklearn.model_selection.cross_val_score) instead of a
#     single train/test split for a more reliable comparison.
# ------------------------------------------------------------------------------

if __name__ == "__main__":
    print("\nDone. Scroll up to compare the models from top to bottom.")
