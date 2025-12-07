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
		<a class="nav-link" href="/projects/1-linear-regression.html">Project page</a> ·
		<a class="nav-link" href="/projects/1-linear-regression/notebook">Rendered notebook</a> ·
		<a class="nav-link" href="/projects/1.%20Linear%20Regression/homeprices.csv">Dataset</a>
	</p>
	<p><em>Status:</em> In Progress</p>
	</div>

<div class="card">
	<h3>Project 2: Multiple Linear Regression</h3>
	<p>Predict home prices using area, bedrooms, and age.</p>
	<p><strong>Concepts:</strong> Multiple Linear Regression, Data Cleaning, Gradient Descent</p>
	<p>
		<a class="nav-link" href="/projects/2-multiple-linear-regression.html">Project page</a> ·
		<a class="nav-link" href="/projects/2-multiple-linear-regression/notebook">Rendered notebook</a> ·
		<a class="nav-link" href="/projects/2.%20Linear%20Regression%20with%20multiple%20features/homeprices%20(1).csv">Dataset</a> ·
		<a class="nav-link" href="/projects/2.%20Linear%20Regression%20with%20multiple%20features/hiring.csv">Dataset</a>
	</p>
	<p><em>Status:</em> In Progress</p>
	</div>

<div class="card">
	<h3>Logistic Regression (Upcoming)</h3>
	<p>Binary and multiclass classification using logistic regression.</p>
	<p><strong>Concepts:</strong> Logistic Regression, Sigmoid Function, Classification</p>
	<p>
		<a class="nav-link" href="/topics/logistic_regression.html">Theory</a>
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
	<h3>Logistic Regression</h3>
	<p>Classification with logistic regression.</p>
	<p><a class="nav-link" href="/projects/logistic-regression.html">Project page</a> · <a class="nav-link" href="/topics/logistic_regression.html">Theory</a></p>
</div>

<div class="card">
	<h3>Multiclass Logistic Regression</h3>
	<p>Softmax regression and OvR strategies.</p>
	<p><a class="nav-link" href="/projects/multiclass-logistic-regression.html">Project page</a></p>
</div>

<div class="card">
	<h3>Decision Trees</h3>
	<p>Tree-based models for classification/regression.</p>
	<p><a class="nav-link" href="/projects/decision-trees.html">Project page</a></p>
</div>

<div class="card">
	<h3>Gradient Descent (Applied)</h3>
	<p>Optimizing models with GD variants.</p>
	<p><a class="nav-link" href="/projects/gradient-descent.html">Project page</a> · <a class="nav-link" href="/topics/gradient_descent.html">Theory</a></p>
</div>

<div class="card">
	<h3>Dummy Variables & One-Hot</h3>
	<p>Encode categorical features for ML.</p>
	<p><a class="nav-link" href="/projects/encoding.html">Project page</a></p>
</div>

<div class="card">
	<h3>Train/Test Split</h3>
	<p>Proper dataset splitting and evaluation.</p>
	<p><a class="nav-link" href="/projects/train-test-split.html">Project page</a></p>
</div>

<div class="card">
	<h3>Save & Load Models</h3>
	<p>Persist models reliably with joblib/pickle.</p>
	<p><a class="nav-link" href="/projects/save-model.html">Project page</a></p>
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
