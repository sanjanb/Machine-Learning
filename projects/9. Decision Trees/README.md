## 1\. What is a Decision Tree? (The Core Idea)

The decision tree algorithm mimics **human decision-making** by breaking down a complex problem into a series of simple, conditional questions [[00:43](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=43)].

  * **The Goal:** To partition the data into subsets that are as **pure** as possible (i.e., subsets where most samples belong to the same class).
  * **When to Use It:** Decision trees are most useful when the decision boundary is **non-linear** or complex, meaning a single straight line (like in Logistic Regression) cannot accurately separate the classes [[00:14](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=14)].
  * **The Structure:**
      * **Root Node:** The starting point, representing the entire dataset.
      * **Internal Nodes:** Represent a feature/attribute test (e.g., "Is Company = Facebook?").
      * **Branches (Edges):** Represent the outcomes of the test.
      * **Leaf Nodes:** The terminal nodes that hold the final decision or prediction (e.g., "Salary \> $100K = Yes").

The process is **iterative**: starting from the root, the data is split based on the chosen feature, and this process repeats on the resulting subsets until a stopping condition is met (e.g., the subset is pure, or a maximum depth is reached).

-----

## 2\. Metrics for Optimal Splitting

The most crucial challenge in building a decision tree is deciding which feature to split on at each step, and in what order [[01:55](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=115)]. This selection is based on which split yields the highest **Information Gain**, which is measured using concepts like Entropy or Gini Impurity.

### A. Entropy (Measure of Randomness)

**Entropy** is a fundamental concept from information theory that measures the **randomness** or **impurity** of a dataset [[02:40](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=160)].

  * **High Entropy (Impure):** If a subset has a mix of classes (e.g., 50% "Yes" and 50% "No"), it is highly random, and its Entropy is close to **1** [[03:06](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=186)].
  * **Low Entropy (Pure):** If a subset contains samples of mostly one class (e.g., 100% "Yes" or 90% "Yes"), it is very pure, and its Entropy is close to **0** [[02:58](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=178)].

### B. Information Gain

The goal of the Decision Tree is to maximize **Information Gain** at every split [[03:33](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=213)].

$$\text{Information Gain} = \text{Entropy}(\text{Parent Node}) - \sum_{i=1}^{n} \frac{\text{Samples}_i}{\text{Total Samples}} \times \text{Entropy}(\text{Child Node}_i)$$

  * The feature that results in the **largest drop** in Entropy (the highest Information Gain) is chosen as the splitting criterion. This ensures the most effective reduction in randomness at each step.

### C. Gini Impurity (Alternative Metric)

**Gini Impurity** (or Gini Index) is an alternative metric to Entropy, often used as the default criterion in scikit-learn [[03:57](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=237)].

  * **Goal:** Like Entropy, it measures the probability of a randomly chosen element being incorrectly classified.
  * **Range:** It ranges from 0 (perfectly pure) to 0.5 (perfectly mixed).
  * **Scikit-learn Default:** By default, the `DecisionTreeClassifier` uses the Gini Impurity criteria to find the best split.

In practice, both Gini Impurity and Entropy often lead to very similar trees, but Gini is computationally slightly faster.

-----

## 3\. Implementation in Scikit-learn

The video demonstrates how to implement a Decision Tree Classifier using Python and scikit-learn.

### A. Data Preparation (Manual Label Encoding)

Since Decision Trees (like most ML models) cannot work directly with text labels, the categorical columns (`Company`, `Job`, `Degree`) are converted to numerical format using **Label Encoding** [[05:44](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=344)].

  * **Process:** The `LabelEncoder` assigns a unique integer (0, 1, 2, etc.) to each unique text value in a column.
  * **Note:** The video uses Label Encoding directly, which is acceptable for tree-based models. Unlike Linear/Logistic Regression, Decision Trees are not sensitive to the false sense of order created by Label Encoding because they work on **ranges** of feature values rather than coefficients.

### B. Model Training

The `DecisionTreeClassifier` is imported from `sklearn.tree` [[09:13](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=553)].

```python
from sklearn.tree import DecisionTreeClassifier

# 1. Initialize the Classifier (uses Gini impurity by default)
model = DecisionTreeClassifier()

# 2. Train the Model (X is features, y is target)
model.fit(inputs_n, target)
```

### C. Model Evaluation and Prediction

  * **Model Score:** The `.score()` method is used to measure the accuracy of the model [[10:46](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=646)]. **Crucially**, the video notes that using the same data for training and testing will result in a misleading **100% accuracy (Score = 1)**, which is an indicator of **overfitting** [[11:01](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=661)]. In a real-world scenario, you must use `train_test_split`.
  * **Prediction:** The `.predict()` method is used to classify new, unseen data points (which must also be passed in their Label Encoded form) [[11:32](http://www.youtube.com/watch?v=PHxYNGo8NcI&t=692)].

The model correctly predicts the survival outcome based on the feature combination, demonstrating the decision tree's ability to classify based on learned rules.

http://googleusercontent.com/youtube_content/17
