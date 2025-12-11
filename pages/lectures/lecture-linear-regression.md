---
layout: default
title: "Lecture: Linear Regression"
permalink: pages/lectures/lecture-linear-regression.html
sidebar: sidebar
---

# Lecture: Linear Regression

This lecture introduces linear regression theory and then applies it via the associated projects.

- Theory: [Linear Regression](/Machine-Learning/topics/linear_regression.html)
- Optimization: [Gradient Descent](/Machine-Learning/topics/gradient_descent.html)
- Cost: [Cost Functions](/Machine-Learning/topics/cost_functions.html)

{% assign lr_projects = site.pages | where: "permalink", "projects/1-linear-regression.html" %}
{% assign mlr_projects = site.pages | where: "permalink", "projects/2-multiple-linear-regression.html" %}
{% assign gd_projects = site.pages | where: "permalink", "projects/gradient-descent.html" %}

{% include lecture_section.html
  title="Applied Projects"
  summary="Practice the theory with hands-on notebooks."
  concepts="Linear regression, multiple variables, gradient descent"
  projects='[
    {"title":"Project 1: Simple Linear Regression","tag":"Regression","category":"regression","difficulty":"beginner","status":"complete","description":"Predict home prices using a single feature.","project_url":"/Machine-Learning/projects/1-linear-regression.html","notebook_url":"/Machine-Learning/projects/1-linear-regression/notebook","colab_url":"https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/1.%20Linear%20Regression/linear-regression%20%281%29.ipynb"},
    {"title":"Project 2: Multiple Linear Regression","tag":"Regression","category":"regression","difficulty":"beginner","status":"complete","description":"Predict home prices using multiple features.","project_url":"/Machine-Learning/projects/2-multiple-linear-regression.html","notebook_url":"/Machine-Learning/projects/2-multiple-linear-regression/notebook","colab_url":"https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/2.%20Linear%20Regression%20with%20multiple%20features/linear-regression-with-multiple-variables.ipynb"},
    {"title":"Project: Gradient Descent","tag":"Optimization","category":"optimization","difficulty":"intermediate","status":"complete","description":"Optimize models via gradient descent.","project_url":"/Machine-Learning/projects/gradient-descent.html","notebook_url":"/Machine-Learning/projects/gradient-descent/notebook","colab_url":"https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/3.%20Gradient%20Descent/gradient-descent-and-cost-function.ipynb"}
  ]'
%}
