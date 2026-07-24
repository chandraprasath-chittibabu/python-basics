# Machine Learning Algorithms — A Simple Guide

A beginner-friendly map of ML: the big categories, the key algorithms in each,
what they're used for, and the core formula behind each one.

---

## The Big Picture: 3 Main Categories

| Category | What it does | You have... | Example question |
|----------|--------------|-------------|------------------|
| **Supervised Learning** | Learns from labeled examples | Inputs **and** known answers | "Is this email spam?" |
| **Unsupervised Learning** | Finds hidden structure | Inputs only, **no** answers | "What customer groups exist?" |
| **Reinforcement Learning** | Learns by trial and reward | An environment + rewards | "How should this robot walk?" |

Supervised learning splits further into:
- **Regression** → predict a *number* (price, temperature)
- **Classification** → predict a *category* (spam/not-spam, cat/dog)

---

## 1. Supervised Learning — Regression (predict a number)

### Linear Regression
- **Idea:** Draw the best straight line through the data.
- **Use cases:** House price prediction, sales forecasting, trend analysis.
- **Formula:**

  $$ y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + \dots + \beta_n x_n $$

  It minimizes the **Mean Squared Error (MSE)**:

  $$ MSE = \frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2 $$

### Polynomial Regression
- **Idea:** Like linear, but fits curves instead of straight lines.
- **Use cases:** Growth curves, non-linear trends.
- **Formula:** $ y = \beta_0 + \beta_1 x + \beta_2 x^2 + \dots + \beta_n x^n $

### Ridge / Lasso Regression (Regularized)
- **Idea:** Linear regression that penalizes large coefficients to avoid overfitting.
- **Use cases:** Many features, risk of overfitting.
- **Formulas (added penalty):**
  - Ridge (L2): $ \text{Loss} = MSE + \lambda \sum \beta_j^2 $
  - Lasso (L1): $ \text{Loss} = MSE + \lambda \sum |\beta_j| $  *(can shrink features to zero → feature selection)*

---

## 2. Supervised Learning — Classification (predict a category)

### Logistic Regression
- **Idea:** Despite the name, it's for classification. Outputs a probability (0–1).
- **Use cases:** Spam detection, disease diagnosis, credit default (yes/no).
- **Formula (sigmoid):**

  $$ P(y=1) = \frac{1}{1 + e^{-(\beta_0 + \beta_1 x_1 + \dots + \beta_n x_n)}} $$

### K-Nearest Neighbors (KNN)
- **Idea:** Classify a point by the majority vote of its *k* closest neighbors.
- **Use cases:** Recommendation, pattern recognition, simple classification.
- **Formula (Euclidean distance):**

  $$ d(p, q) = \sqrt{\sum_{i=1}^{n}(p_i - q_i)^2} $$

### Decision Tree
- **Idea:** A flowchart of yes/no questions that splits data into groups.
- **Use cases:** Loan approval, medical decisions, easy-to-explain models.
- **Formula (Gini impurity — how "mixed" a group is):**

  $$ Gini = 1 - \sum_{i=1}^{c} p_i^2 $$

### Random Forest
- **Idea:** Many decision trees vote together ("wisdom of the crowd"). More accurate and stable than one tree.
- **Use cases:** Fraud detection, feature importance, general-purpose tasks.
- **How:** Builds many trees on random subsets of data/features, then averages (regression) or votes (classification).

### Support Vector Machine (SVM)
- **Idea:** Find the boundary (hyperplane) that separates classes with the widest gap (margin).
- **Use cases:** Text classification, image recognition, small-to-medium datasets.
- **Formula (decision boundary):** $ w \cdot x + b = 0 $, maximizing margin $ \frac{2}{\|w\|} $.

### Naive Bayes
- **Idea:** Uses probability (Bayes' theorem) and assumes features are independent.
- **Use cases:** Spam filtering, sentiment analysis, document classification.
- **Formula (Bayes' theorem):**

  $$ P(A|B) = \frac{P(B|A)\,P(A)}{P(B)} $$

---

## 3. Unsupervised Learning (find structure, no labels)

### K-Means Clustering
- **Idea:** Group data into *k* clusters by closeness to a cluster center.
- **Use cases:** Customer segmentation, image compression, grouping documents.
- **Formula (minimize within-cluster distance):**

  $$ J = \sum_{j=1}^{k}\sum_{i \in C_j} \|x_i - \mu_j\|^2 $$

  where $ \mu_j $ is the center of cluster $ j $.

### Hierarchical Clustering
- **Idea:** Build a tree of nested clusters (merge closest pairs step by step).
- **Use cases:** Taxonomy building, gene analysis, dendrograms.

### Principal Component Analysis (PCA)
- **Idea:** Reduce many features into a few that capture most information.
- **Use cases:** Dimensionality reduction, visualization, noise reduction, speeding up models.
- **How:** Finds directions (eigenvectors of the covariance matrix) of maximum variance.

### Association Rules (Apriori)
- **Idea:** Find "if this, then that" patterns in transactions.
- **Use cases:** Market basket analysis ("people who buy bread also buy butter").
- **Formulas:**
  - Support: $ \frac{\text{transactions with A and B}}{\text{total}} $
  - Confidence: $ \frac{\text{transactions with A and B}}{\text{transactions with A}} $

---

## 4. Reinforcement Learning (learn by reward)

### Q-Learning
- **Idea:** An agent learns which actions give the most reward over time.
- **Use cases:** Game AI, robotics, self-driving, trading bots.
- **Formula (Q-value update):**

  $$ Q(s,a) \leftarrow Q(s,a) + \alpha \left[ r + \gamma \max_{a'} Q(s',a') - Q(s,a) \right] $$

  - $ \alpha $ = learning rate, $ \gamma $ = discount factor (how much future rewards matter), $ r $ = reward.

---

## 5. Neural Networks & Deep Learning (a special powerful family)

- **Idea:** Layers of connected "neurons" that learn complex patterns. Powers modern AI.
- **Types & use cases:**
  - **CNN** (Convolutional) → images, video (face recognition, medical imaging)
  - **RNN / LSTM** → sequences (text, speech, time series)
  - **Transformers** → language (ChatGPT, translation) — today's state of the art
- **Core building block (one neuron):**

  $$ output = f\left(\sum_{i} w_i x_i + b\right) $$

  where $ f $ is an activation function (e.g. ReLU: $ f(x) = \max(0, x) $).

---

## Quick Cheat Sheet: "Which algorithm do I use?"

| Your goal | Try this |
|-----------|----------|
| Predict a number | Linear Regression, Random Forest |
| Yes/No or category | Logistic Regression, Random Forest, SVM |
| Group similar things (no labels) | K-Means, Hierarchical Clustering |
| Too many features | PCA |
| "Bought together" patterns | Apriori |
| Images | CNN |
| Text / language | Transformers, Naive Bayes |
| Learn by trial and error | Q-Learning |

---

## How to Judge a Model (common metrics)

**Regression:**
- MSE / RMSE — average error (lower = better)
- R² — how much variance is explained (closer to 1 = better)

**Classification:**
- Accuracy — % correct
- Precision — of predicted positives, how many were right
- Recall — of actual positives, how many were caught
- F1 Score — balance of precision & recall: $ 2 \times \frac{P \times R}{P + R} $

---

*Tip: Start simple (Linear/Logistic Regression), measure performance, then move to
more complex models (Random Forest, Neural Nets) only if you need better accuracy.*
