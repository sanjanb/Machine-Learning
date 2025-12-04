## 1. The Fundamental Problem: Overfitting and Bias

The video starts by explaining why you should **not** train a model using the entire dataset [[00:16](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=16)].

* **The Goal of Training:** To build a model (or a "brain") that generalizes well to new, unseen data.
* **The Flawed Strategy:** If you train a model on all the data and test it on the same data, the model will likely achieve a very high accuracy score (e.g., 99% or 100%).
* **The Reality (Overfitting):** This high score is often misleading. The model has simply **memorized** the training examples, including their noise and outliers, instead of learning the underlying pattern. This phenomenon is called **overfitting**. A memorized model will perform poorly on *new* data.
* **The Solution:** You must use samples for testing that the model has **not seen before** to get an honest assessment of its accuracy and generalization ability [[00:31](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=31)].

***

## 2. The Solution: Training and Testing Split

The correct strategy is to divide the original dataset into two mutually exclusive sets:

### A. Training Set ($\mathbf{X}_{\text{train}}, \mathbf{Y}_{\text{train}}$)
* Used to **train** (fit) the machine learning model. This is where the model learns the relationships between the features ($\mathbf{X}$) and the target ($\mathbf{Y}$) [[04:58](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=298)].

### B. Testing Set ($\mathbf{X}_{\text{test}}, \mathbf{Y}_{\text{test}}$)
* Used **only** to **evaluate** the trained model. Since the model has never encountered this data during training, the accuracy score here is a reliable measure of the model's performance on real-world data [[05:09](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=309)].

***

## 3. Implementing `train_test_split` with Scikit-learn

The video demonstrates the use of the `train_test_split` function from the `sklearn.model_selection` module [[02:16](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=136)].

| Component | Code | Explanation |
| :--- | :--- | :--- |
| **Import** | `from sklearn.model_selection import train_test_split` | Imports the necessary splitting function. |
| **Call** | `train_test_split(X, Y, test_size=0.2)` | Takes the full features ($\mathbf{X}$) and targets ($\mathbf{Y}$) as input. |
| **Output** | `X_train, X_test, Y_train, Y_test` | Returns **four** resulting data arrays or subsets. |

### Key Parameters:

| Parameter | Purpose | Example |
| :--- | :--- | :--- |
| **`test_size`** | Specifies the ratio of the total dataset to be reserved for the testing set [[02:40](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=160)]. | `test_size=0.2` means **20%** of the data is for testing, and the remaining 80% is for training. |
| **`random_state`** | Controls the **randomness** of the data split [[03:53](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=233)]. | `random_state=10` |

### The `random_state` Parameter: Ensuring Reproducibility
By default, `train_test_split` selects samples **randomly** [[03:39](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=219)]. If you run the code multiple times, the exact rows assigned to $\mathbf{X}_{\text{train}}$ and $\mathbf{X}_{\text{test}}$ will change.

* By setting `random_state` to any fixed integer (e.g., 10, 42, 0), you ensure that the same random set of rows is selected every time the code runs [[04:02](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=242)]. This is crucial for **reproducibility** when sharing code or debugging.

***

## 4. Model Evaluation using `.score()`

Once the model is trained using **only** $\mathbf{X}_{\text{train}}$ and $\mathbf{Y}_{\text{train}}$, its accuracy is checked using the `.score()` method [[05:38](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=338)].

* **Prediction:** The model uses $\mathbf{X}_{\text{test}}$ to make predictions.
* **Comparison:** The predicted values are compared against the actual known values, $\mathbf{Y}_{\text{test}}$.
* **Result:** For a regression model (like the Linear Regression used in the video), the `.score()` method returns the **coefficient of determination, R-squared ($R^2$)**.

The video shows a result of **89%** ($0.89$), which represents the model's accuracy on the unseen data [[05:55](http://www.youtube.com/watch?v=fwY9Qv96DJY&t=355)]. This is a true measure of the model's ability to generalize, unlike a score calculated on the training set.


http://googleusercontent.com/youtube_content/15
