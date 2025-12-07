---
title: Logistic Regression
keywords: cs229, machine learning, logistic regression
sidebar: sidebar
permalink: topics/logistic_regression.html
layout: default
folder: topics
---

## Logistic Regression

This section covers Logistic Regression for classification tasks.

### Key Concepts

- Sigmoid function
- Cost function for logistic regression
- Gradient descent
- Multiclass classification

### Mathematical Foundation

Sigmoid function:
$$g(z) = \frac{1}{1 + e^{-z}}$$

Hypothesis function:
$$h_\theta(x) = g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}$$

Cost function:
$$J(\theta) = \frac{1}{m} \sum_{i=1}^{m} [-y^{(i)} \log(h_\theta(x^{(i)})) - (1-y^{(i)}) \log(1-h_\theta(x^{(i)}))]$$

### Practical Implementation

📊 **See this concept in action:** [Project 7: Logistic Regression](/Machine-Learning/projects/logistic-regression.html) and [Project 8: Multiclass Logistic Regression](/Machine-Learning/projects/multiclass-logistic-regression.html)

### Applications

- Email spam detection
- Medical diagnosis
- Marketing response prediction

_Detailed notes to be added based on project learnings._
