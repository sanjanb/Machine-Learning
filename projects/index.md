---
title: "Projects"
keywords: cs229, machine learning, projects
sidebar: sidebar
permalink: projects/
layout: default
---

# Machine Learning Projects

This section contains practical projects that implement various machine learning concepts and algorithms.

## Current Projects

### Project 1: Simple Linear Regression

## Current Projects

<div class="cards">

<div class="card">
	<h3>Project 1: Simple Linear Regression</h3>
	<p>Predict home prices using a single feature (area).</p>
	<p><strong>Concepts:</strong> Linear Regression, Gradient Descent, Cost Functions</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/1-linear-regression.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/1-linear-regression/notebook">Rendered notebook</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/1.%20Linear%20Regression/homeprices.csv">Dataset</a>
	</p>
	<p><em>Status:</em> In Progress</p>
	</div>

<div class="card">
	<h3>Project 2: Multiple Linear Regression</h3>
	<p>Predict home prices using area, bedrooms, and age.</p>
	<p><strong>Concepts:</strong> Multiple Linear Regression, Data Cleaning, Gradient Descent</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/2-multiple-linear-regression.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/2-multiple-linear-regression/notebook">Rendered notebook</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/2.%20Linear%20Regression%20with%20multiple%20features/homeprices%20(1).csv">Dataset 1</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/2.%20Linear%20Regression%20with%20multiple%20features/hiring.csv">Dataset 2</a>
	</p>
	<p><em>Status:</em> In Progress</p>
	</div>

<div class="card">
	<h3>Logistic Regression (Upcoming)</h3>
	<p>Binary and multiclass classification using logistic regression.</p>
	<p><strong>Concepts:</strong> Logistic Regression, Sigmoid Function, Classification</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/topics/logistic_regression.html">Theory</a>
	</p>
	<p><em>Status:</em> Planned</p>
	</div>

</div>

<style>
.cards {
	display: grid;
	grid-template-columns: repeat(auto-fill,minmax(280px,1fr));
	gap: 1rem;
}
.card {
	border: 1px solid #bdc3c7;
	border-radius: 8px;
	padding: 1rem;
	background: #fff;
	box-shadow: 0 2px 10px rgba(0,0,0,0.05);
}
.card h3 { margin: 0 0 0.5rem 0; }
.card p { margin: 0.25rem 0; }
</style>

## More Projects

<div class="cards">

<div class="card">
	<h3>Project 3: Gradient Descent</h3>
	<p>Optimizing models with gradient descent variants.</p>
	<p><strong>Concepts:</strong> Batch/Stochastic GD, Learning Rate, Convergence</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/gradient-descent.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/gradient-descent/notebook">Rendered notebook</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/3.%20Gradient%20Descent/test_scores.csv">Dataset</a>
	</p>
</div>

<div class="card">
	<h3>Project 4: Save & Load Models</h3>
	<p>Persist trained models with joblib/pickle.</p>
	<p><strong>Concepts:</strong> Serialization, Model Persistence, Reproducibility</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/save-model.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/save-model/notebook">Rendered notebook</a>
	</p>
</div>

<div class="card">
	<h3>Project 5: Dummy Variables & One-Hot</h3>
	<p>Encode categorical features for ML models.</p>
	<p><strong>Concepts:</strong> One-Hot Encoding, Drop-First, Multicollinearity</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/encoding.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/encoding/notebook">Rendered notebook</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/5.%20Dummy%20variable%20%26%20one-hot%20encoding/homeprices%20(2).csv">Dataset 1</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/5.%20Dummy%20variable%20%26%20one-hot%20encoding/carprices.csv">Dataset 2</a>
	</p>
</div>

<div class="card">
	<h3>Project 6: Train/Test Split</h3>
	<p>Proper dataset splitting and evaluation.</p>
	<p><strong>Concepts:</strong> Random State, Stratification, Data Leakage</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/train-test-split.html">Project page</a>
	</p>
	<p><em>Status:</em> Notebook Coming Soon</p>
</div>

<div class="card">
	<h3>Project 7: Logistic Regression</h3>
	<p>Binary classification with logistic regression.</p>
	<p><strong>Concepts:</strong> Sigmoid, Decision Boundary, Classification</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/logistic-regression.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/logistic-regression/notebook">Rendered notebook</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/7.%20Logistic%20Regression/insurance_data.csv">Dataset 1</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/7.%20Logistic%20Regression/HR_comma_sep.csv">Dataset 2</a>
	</p>
</div>

<div class="card">
	<h3>Project 8: Multiclass Logistic Regression</h3>
	<p>Softmax regression and OvR strategies.</p>
	<p><strong>Concepts:</strong> Softmax, Cross-Entropy, One-vs-Rest</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/multiclass-logistic-regression.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/multiclass-logistic-regression/notebook">Rendered notebook</a>
	</p>
</div>

<div class="card">
	<h3>Project 9: Decision Trees</h3>
	<p>Tree-based models for classification/regression.</p>
	<p><strong>Concepts:</strong> Gini, Entropy, Information Gain, Pruning</p>
	<p>
		<a class="nav-link" href="/Machine-Learning/projects/decision-trees.html">Project page</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/decision-trees/notebook">Main notebook</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/decision-trees/exercise/notebook">Exercise</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/9.%20Decision%20Trees/salaries.csv">Dataset 1</a> ·
		<a class="nav-link" href="/Machine-Learning/projects/9.%20Decision%20Trees/Exercise/titanic.csv">Dataset 2</a>
	</p>
</div>

</div>

### Project 2: Multiple Linear Regression

**Status:** In Progress  
**Concepts Used:** Multiple Linear Regression, Data Cleaning, Gradient Descent  
**Description:** Predicts home prices using area, bedrooms, and age. Includes data preprocessing and model evaluation.

### Project: Classification with Logistic Regression

**Status:** Planned  
**Concepts Used:** Logistic Regression, Sigmoid Function, Classification  
**Description:** Binary and multiclass classification using logistic regression.

---

## Project Structure

Each project includes:

- **📝 Problem Statement** - What we're trying to solve
- **🔧 Implementation** - Step-by-step code with explanations
- **📊 Results & Analysis** - Visualizations and performance metrics
- **🔗 Concept Links** - Connections to theory pages
- **💡 Key Learnings** - Insights and takeaways

---

## Quick Links to Related Concepts

| Concept             | Theory Page                                   | Used in Projects |
| ------------------- | --------------------------------------------- | ---------------- |
| Linear Regression   | [📖 Theory](/topics/linear_regression.html)   | Project 1        |
| Logistic Regression | [📖 Theory](/topics/logistic_regression.html) | Project 2        |
| Gradient Descent    | [📖 Theory](/topics/gradient_descent.html)    | Projects 1, 2, 3 |
| Cost Functions      | [📖 Theory](/topics/cost_functions.html)      | Projects 1, 2, 3 |

---

## 🚀 Getting Started

1. Choose a project that matches your current learning level
2. Review the related concept pages first
3. Follow the implementation step-by-step
4. Experiment with the code and parameters
5. Document your findings and insights

_New projects are added regularly as we progress through the course!_
