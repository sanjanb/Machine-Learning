---
title: Gradient Descent
keywords: cs229, machine learning, gradient descent, optimization
sidebar: sidebar
permalink: topics/gradient_descent.html
layout: default
folder: topics
---

## Gradient Descent

Gradient descent is an optimization algorithm used to minimize cost functions in machine learning.

### Key Concepts

- Learning rate (α)
- Partial derivatives
- Local vs global minima
- Convergence criteria

### Mathematical Foundation

Update rule:
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$

Where:

- $\theta_j$ = parameter j
- $\alpha$ = learning rate
- $J(\theta)$ = cost function

### Types of Gradient Descent

1. **Batch Gradient Descent** - Uses entire dataset
2. **Stochastic Gradient Descent (SGD)** - Uses one sample at a time
3. **Mini-batch Gradient Descent** - Uses small batches

### Practical Implementation

📊 **See this concept in action:**

- [Project 1: Linear Regression Analysis](/projects/project1-linear-regression.html)
- [Project 2: Logistic Regression](/projects/project2-logistic-regression.html) _(Coming Soon)_

### Learning Rate Selection

- **Too small:** Slow convergence
- **Too large:** May overshoot minimum
- **Just right:** Efficient convergence

_Detailed notes and visualizations to be added based on project implementations._
