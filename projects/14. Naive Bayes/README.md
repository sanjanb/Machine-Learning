## Part 1: Theoretical Foundation of Naive Bayes

The video begins by establishing the core mathematical concept that underlies Naive Bayes: **Conditional Probability** and **Bayes' Theorem**.

### 1. Basic Probability and Conditional Probability [[00:18](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=18)]

* **Basic Probability:** The likelihood of an event occurring (e.g., getting a Queen from a deck of cards) is calculated as (Number of favorable outcomes) / (Total possible outcomes).
* **Conditional Probability:** This is the probability of an event (A) occurring **given that another event (B) has already occurred**.
    * **Notation:** $P(A|B)$ (read as "the probability of A given B").
    * **Example (Cards):** What is the probability of picking a Queen, *given* that the card you picked is a Diamond? Since you now only consider the 13 Diamond cards, the probability is $1/13$.

### 2. Bayes' Theorem [[01:50](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=110)]

Thomas Bayes' famous theorem provides a mathematical way to calculate conditional probability, especially useful when you know the reverse conditional probability.

* **The Equation:**
    $$P(A|B) = \frac{P(B|A) \cdot P(A)}{P(B)}$$
* **Interpretation in Classification:**
    * **$P(A|B)$ (Posterior Probability):** The probability of the target class $A$ (e.g., *Survival*) given the new input features $B$ (e.g., *Male, Pclass 3*). This is what we want to find.
    * **$P(B|A)$ (Likelihood):** The probability of observing feature $B$ (e.g., *Male*) given that the target class $A$ has occurred (e.g., *Survival*). This is calculated from the training data.
    * **$P(A)$ (Prior Probability):** The overall probability of the target class $A$ (e.g., *Overall Survival Rate*).
    * **$P(B)$ (Evidence):** The overall probability of observing the features $B$.

### 3. The "Naive" Assumption [[03:22](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=202)]

The reason the algorithm is called **Naive Bayes** is due to a simplifying assumption it makes:

* **Assumption:** The algorithm assumes that all features (e.g., Sex, Pclass, Age, Fare) are **independent** of each other.
* **Reality vs. Naivety:** In reality, features are often dependent (e.g., `Fare` and `Cabin` are related, as higher fares buy better cabins).
* **Benefit:** This "naive" independence assumption significantly **simplifies the calculation** and is the reason the algorithm is computationally very fast, yet often highly effective, especially for text classification.

### 4. Common Use Cases for Naive Bayes [[04:21](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=261)]

Naive Bayes is historically very popular for:
* Email Spam Detection (Classifying text based on words).
* Handwriting Character Recognition.
* Weather Prediction.
* News Article Categorization.
* Face Detection.

***

## Part 2: Titanic Survival Prediction (Coding Implementation)

The video uses a real-world dataset (Titanic survival) to demonstrate how to prepare data and train a **Gaussian Naive Bayes** model.

### 1. Data Cleaning and Preprocessing [[04:38](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=278)]

Before training, the raw data must be cleaned and transformed into a numeric format:

* **Feature Selection (Drop Irrelevant Columns):** Columns deemed irrelevant to survival (e.g., `Name`, `Ticket`, `Cabin`, `PassengerId`) are dropped to simplify the model. [[05:05](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=305)]
* **Separate Features and Target:**
    * **Target (`y`):** `Survived` column is isolated.
    * **Inputs (`X`):** The remaining features are used as inputs. [[05:25](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=325)]
* **One-Hot Encoding (Handling Text):** The `Sex` column (containing 'male'/'female') is a categorical text feature.
    * Machine learning models cannot handle text, so it must be converted to numbers using **one-hot encoding** (Pandas' `pd.get_dummies` function). [[06:01](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=361)]
    * This creates new numeric columns (`Sex_male`, `Sex_female`) and the original `Sex` column is dropped.
* **Handling Missing Values (Imputation):** The `Age` column often contains missing values (`NaN`).
    * The missing values are **imputed** (filled) with the **mean** (average) value of the entire `Age` column. [[07:21](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=441)]

### 2. Model Training and Evaluation [[08:47](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=527)]

#### A. Data Splitting
* The clean dataset is split into **Training** (80%) and **Testing** (20%) samples using Scikit-learn's `train_test_split`.
* This ensures the model is trained on one set of data (`X_train`, `y_train`) and evaluated on completely unseen data (`X_test`, `y_test`), preventing bias.

#### B. Naive Bayes Model Selection [[10:06](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=606)]
* The video uses the **`GaussianNB`** class from Scikit-learn.
* **Gaussian Assumption:** This is the appropriate choice when features are assumed to follow a **Gaussian (Normal or Bell Curve) distribution** (which is often a reasonable assumption for features like Age and Fare).

#### C. Training and Scoring
* **Training:** The model is trained by calling the `fit` method: `model.fit(X_train, y_train)` [[10:36](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=636)].
* **Evaluation:** Model accuracy is measured using the `score` method, which returns the percentage of correct predictions on the test set (e.g., 77%-81%). [[11:06](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=666)]

#### D. Prediction and Probability [[12:15](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=735)]
* **Prediction:** The `model.predict(X_test)` function returns the final class label (0 for not survived, 1 for survived).
* **Predict Probability:** The `model.predict_proba(X_test)` function returns the **posterior probability** for each class, demonstrating the core Bayesian output.
    * Example: An output of `[0.97, 0.03]` means there is a **97% probability of not surviving** (Class 0) and a 3% probability of surviving (Class 1). The model then chooses the class with the highest probability (0). [[12:43](http://www.youtube.com/watch?v=PPeaRc-r1OI&t=763)]



http://googleusercontent.com/youtube_content/2
