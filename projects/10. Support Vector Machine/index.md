---
layout: default
title: "Project: Support Vector Machine"
permalink: projects/support-vector-machine.html
sidebar: sidebar
image: /Machine-Learning/assets/images/og-default.svg
---

# Project 10: Support Vector Machine (SVM)

This project explores Support Vector Machines for classification tasks, including kernel tricks and hyperparameter tuning.

## Summary

- Goal: Classify data using optimal hyperplanes with maximum margin
- Concepts: Support vectors, kernels (linear, RBF), C parameter, gamma tuning
- Dataset: Iris dataset for multiclass classification
- Pipeline: Data visualization → SVM training → Hyperparameter optimization → Evaluation

## Notebook

- [Rendered notebook](/Machine-Learning/projects/support-vector-machine/notebook)
- Open in Colab: [Launch](/redirect?target=https%3A%2F%2Fcolab.research.google.com%2Fgithub%2Fsanjanb%2FMachine-Learning%2Fblob%2Fmain%2Fprojects%2F10.%2520Support%2520Vector%2520Machine%2Fsupport-vector-machine.ipynb)

## Key Concepts

### Support Vectors

The critical data points closest to the decision boundary that define the optimal hyperplane.

### Kernel Trick

Transforms data into higher dimensions to make non-linearly separable data linearly separable:

- **Linear Kernel:** For linearly separable data
- **RBF (Radial Basis Function):** For non-linear patterns
- **Polynomial Kernel:** For polynomial decision boundaries

### Hyperparameters

- **C (Regularization):** Controls the trade-off between maximizing margin and minimizing classification errors
  - High C: Strict (may overfit)
  - Low C: More tolerant (may underfit)
- **Gamma:** Defines the influence of single training examples
  - High gamma: Close influence (may overfit)
  - Low gamma: Far-reaching influence (smoother decision boundary)

## Applications

- Image classification
- Text categorization
- Bioinformatics (protein classification)
- Handwriting recognition

## Related Topics

- [Logistic Regression](/Machine-Learning/topics/logistic_regression.html)
- [Cost Functions](/Machine-Learning/topics/cost_functions.html)

## Key Learnings

{% include_relative README.md %}
