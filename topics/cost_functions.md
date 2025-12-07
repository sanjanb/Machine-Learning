---
title: Cost Functions
keywords: cs229, machine learning, cost functions, loss functions
sidebar: sidebar
permalink: topics/cost_functions.html
layout: default
folder: topics
---

## Cost Functions

Cost functions (also called loss functions) measure how well our model predictions match the actual values.

### Key Concepts

- Mean Squared Error (MSE)
- Cross-entropy loss
- Regularization
- Bias-variance tradeoff

### Common Cost Functions

#### 1. Mean Squared Error (Regression)

$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

#### 2. Logistic Loss (Classification)

$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)} \log(h_\theta(x^{(i)})) - (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$$

#### 3. Regularized Cost Functions

**L1 Regularization (Lasso):**
$$J(\theta) = \text{Original Cost} + \lambda \sum_{j=1}^{n} |\theta_j|$$

**L2 Regularization (Ridge):**
$$J(\theta) = \text{Original Cost} + \lambda \sum_{j=1}^{n} \theta_j^2$$

### Practical Implementation

📊 **See this concept in action:**

- [Project 1: Simple Linear Regression](/Machine-Learning/projects/1-linear-regression.html)
- [Project 3: Gradient Descent](/Machine-Learning/projects/gradient-descent.html)
- [Project 2: Logistic Regression](/projects/project2-logistic-regression.html) _(Coming Soon)_

### Choosing the Right Cost Function

| Problem Type               | Recommended Cost Function |
| -------------------------- | ------------------------- |
| Regression                 | Mean Squared Error (MSE)  |
| Binary Classification      | Logistic Loss             |
| Multi-class Classification | Cross-entropy Loss        |
| Sparse Features            | L1 Regularized            |
| Prevent Overfitting        | L2 Regularized            |

_Detailed examples and visualizations to be added based on project implementations._
