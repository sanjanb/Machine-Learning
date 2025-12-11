---
title: "Projects"
keywords: cs229, machine learning, projects
sidebar: sidebar
permalink: projects/
layout: default
image: /Machine-Learning/assets/images/og-default.svg
---

<section class="hero">
	<div class="hero-content">
		<h1>Machine Learning Projects</h1>
		<p>Hands-on implementations of core ML concepts with clean notebooks, datasets, and concise write-ups.</p>
		<div class="hero-actions">
			<a class="btn primary" href="/Machine-Learning/topics/linear_regression.html">Start with Linear Regression</a>
			<a class="btn" href="/Machine-Learning/projects/support-vector-machine.html">Explore SVM</a>
		</div>
	</div>
</section>

<section class="filters" aria-label="Project filters">
	<div class="filter-grid">
		<label class="filter">
			<span>Search</span>
			<input id="search-input" type="search" placeholder="Search projects..." aria-label="Search projects" />
		</label>
		<label class="filter">
			<span>Category</span>
			<select id="filter-category">
				<option value="all" selected>All</option>
				<option value="regression">Regression</option>
				<option value="classification">Classification</option>
				<option value="preprocessing">Preprocessing</option>
				<option value="optimization">Optimization</option>
			</select>
		</label>
		<label class="filter">
			<span>Difficulty</span>
			<select id="filter-difficulty">
				<option value="all" selected>All</option>
				<option value="beginner">Beginner</option>
				<option value="intermediate">Intermediate</option>
				<option value="advanced">Advanced</option>
			</select>
		</label>
		<label class="filter">
			<span>Status</span>
			<select id="filter-status">
				<option value="all" selected>All</option>
				<option value="complete">Complete</option>
				<option value="in-progress">In Progress</option>
				<option value="planned">Planned</option>
			</select>
		</label>
	</div>
</section>

<section class="cards" id="projects-grid">
	<!-- Project 1 -->
	<article class="card" data-category="regression" data-difficulty="beginner" data-status="complete">
		<header>
			<h3>Project 1: Simple Linear Regression</h3>
			<span class="tag">Regression</span>
		</header>
		<p>Predict home prices using a single feature (area).</p>
		<p><strong>Concepts:</strong> Linear Regression, Gradient Descent, Cost Functions</p>
		<p class="links">
			<a class="nav-link" href="/Machine-Learning/projects/1-linear-regression.html">Project page</a>
			<span>·</span>
			<a class="nav-link" href="/Machine-Learning/projects/1-linear-regression/notebook">Notebook</a>
			<span>·</span>
			<a class="nav-link" href="/Machine-Learning/projects/1.%20Linear%20Regression/homeprices.csv">Dataset</a>
		</p>
	</article>

    <!-- Project 2 -->
    <article class="card" data-category="regression" data-difficulty="beginner" data-status="complete">
    	<header>
    		<h3>Project 2: Multiple Linear Regression</h3>
    		<span class="tag">Regression</span>
    	</header>
    	<p>Predict home prices using area, bedrooms, and age.</p>
    	<p><strong>Concepts:</strong> Multiple Linear Regression, Data Cleaning, Gradient Descent</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/2-multiple-linear-regression.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/2-multiple-linear-regression/notebook">Notebook</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/2.%20Linear%20Regression%20with%20multiple%20features/homeprices%20(1).csv">Dataset</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/2.%20Linear%20Regression%20with%20multiple%20features/hiring.csv">Dataset</a>
    	</p>
    </article>

    <!-- Project 3 -->
    <article class="card" data-category="optimization" data-difficulty="intermediate" data-status="complete">
    	<header>
    		<h3>Project 3: Gradient Descent</h3>
    		<span class="tag">Optimization</span>
    	</header>
    	<p>Optimizing models with gradient descent variants.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/gradient-descent.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/gradient-descent/notebook">Notebook</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/3.%20Gradient%20Descent/test_scores.csv">Dataset</a>
    	</p>
    </article>

    <!-- Project 4 -->
    <article class="card" data-category="preprocessing" data-difficulty="beginner" data-status="complete">
    	<header>
    		<h3>Project 4: Save & Load Models</h3>
    		<span class="tag">Utility</span>
    	</header>
    	<p>Persist trained models with joblib/pickle.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/save-model.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/save-model/notebook">Notebook</a>
    	</p>
    </article>

    <!-- Project 5 -->
    <article class="card" data-category="preprocessing" data-difficulty="beginner" data-status="complete">
    	<header>
    		<h3>Project 5: Dummy Variables & One-Hot</h3>
    		<span class="tag">Preprocessing</span>
    	</header>
    	<p>Encode categorical features for ML models.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/encoding.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/encoding/notebook">Notebook</a>
				{% include project_card.html
				  title="Project 1: Simple Linear Regression"
				  tag="Regression"
				  category="regression"
				  difficulty="beginner"
				  status="complete"
				  description="Predict home prices using a single feature (area)."
				  concepts="Linear Regression, Gradient Descent, Cost Functions"
				  project_url="/Machine-Learning/projects/1-linear-regression.html"
				  notebook_url="/Machine-Learning/projects/1-linear-regression/notebook"
				  dataset_urls="/Machine-Learning/projects/1.%20Linear%20Regression/homeprices.csv::Dataset"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/1.%20Linear%20Regression/linear-regression%20%281%29.ipynb"
				%}

				{% include project_card.html
				  title="Project 2: Multiple Linear Regression"
				  tag="Regression"
				  category="regression"
				  difficulty="beginner"
				  status="complete"
				  description="Predict home prices using area, bedrooms, and age."
				  concepts="Multiple Linear Regression, Data Cleaning, Gradient Descent"
				  project_url="/Machine-Learning/projects/2-multiple-linear-regression.html"
				  notebook_url="/Machine-Learning/projects/2-multiple-linear-regression/notebook"
				  dataset_urls="/Machine-Learning/projects/2.%20Linear%20Regression%20with%20multiple%20features/homeprices%20(1).csv::Dataset|/Machine-Learning/projects/2.%20Linear%20Regression%20with%20multiple%20features/hiring.csv::Dataset"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/2.%20Linear%20Regression%20with%20multiple%20features/linear-regression-with-multiple-variables.ipynb"
				%}

				{% include project_card.html
				  title="Project 3: Gradient Descent"
				  tag="Optimization"
				  category="optimization"
				  difficulty="intermediate"
				  status="complete"
				  description="Optimizing models with gradient descent variants."
				  project_url="/Machine-Learning/projects/gradient-descent.html"
				  notebook_url="/Machine-Learning/projects/gradient-descent/notebook"
				  dataset_urls="/Machine-Learning/projects/3.%20Gradient%20Descent/test_scores.csv::Dataset"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/3.%20Gradient%20Descent/gradient-descent-and-cost-function.ipynb"
				%}

				{% include project_card.html
				  title="Project 4: Save & Load Models"
				  tag="Utility"
				  category="preprocessing"
				  difficulty="beginner"
				  status="complete"
				  description="Persist trained models with joblib/pickle."
				  project_url="/Machine-Learning/projects/save-model.html"
				  notebook_url="/Machine-Learning/projects/save-model/notebook"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/4.%20Save%20the%20model/save-model-using-joblib-and-pick-le.ipynb"
				%}

				{% include project_card.html
				  title="Project 5: Dummy Variables & One-Hot"
				  tag="Preprocessing"
				  category="preprocessing"
				  difficulty="beginner"
				  status="complete"
				  description="Encode categorical features for ML models."
				  project_url="/Machine-Learning/projects/encoding.html"
				  notebook_url="/Machine-Learning/projects/encoding/notebook"
				  dataset_urls="/Machine-Learning/projects/5.%20Dummy%20variable%20%26%20one-hot%20encoding/homeprices%20(2).csv::Dataset|/Machine-Learning/projects/5.%20Dummy%20variable%20%26%20one-hot%20encoding/carprices.csv::Dataset"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/5.%20Dummy%20variable%20%26%20one-hot%20encoding/dummy-variable-and-one-hot-encoding.ipynb"
				%}

				{% include project_card.html
				  title="Project 6: Train/Test Split"
				  tag="Evaluation"
				  category="preprocessing"
				  difficulty="beginner"
				  status="planned"
				  description="Proper dataset splitting and evaluation."
				  project_url="/Machine-Learning/projects/train-test-split.html"
				%}

				{% include project_card.html
				  title="Project 7: Logistic Regression"
				  tag="Classification"
				  category="classification"
				  difficulty="beginner"
				  status="complete"
				  description="Binary classification with logistic regression."
				  project_url="/Machine-Learning/projects/logistic-regression.html"
				  notebook_url="/Machine-Learning/projects/logistic-regression/notebook"
				  dataset_urls="/Machine-Learning/projects/7.%20Logistic%20Regression/insurance_data.csv::Dataset|/Machine-Learning/projects/7.%20Logistic%20Regression/HR_comma_sep.csv::Dataset"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/7.%20Logistic%20Regression/logistic-regression.ipynb"
				%}

				{% include project_card.html
				  title="Project 8: Multiclass Logistic Regression"
				  tag="Classification"
				  category="classification"
				  difficulty="intermediate"
				  status="complete"
				  description="Softmax regression and OvR strategies."
				  project_url="/Machine-Learning/projects/multiclass-logistic-regression.html"
				  notebook_url="/Machine-Learning/projects/multiclass-logistic-regression/notebook"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/8.%20Multi%20Class%20Logistic%20Regression/multi-classlogistic-regression.ipynb"
				%}

				{% include project_card.html
				  title="Project 9: Decision Trees"
				  tag="Classification"
				  category="classification"
				  difficulty="intermediate"
				  status="complete"
				  description="Tree-based models for classification/regression."
				  project_url="/Machine-Learning/projects/decision-trees.html"
				  notebook_url="/Machine-Learning/projects/decision-trees/notebook"
				  exercise_url="/Machine-Learning/projects/decision-trees/exercise/notebook"
				  exercise_label="Exercise"
				  dataset_urls="/Machine-Learning/projects/9.%20Decision%20Trees/salaries.csv::Dataset|/Machine-Learning/projects/9.%20Decision%20Trees/Exercise/titanic.csv::Dataset"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/9.%20Decision%20Trees/decision-tree.ipynb"
				%}

				{% include project_card.html
				  title="Project 10: Support Vector Machine"
				  tag="Classification"
				  category="classification"
				  difficulty="intermediate"
				  status="complete"
				  description="Classification with optimal hyperplanes and kernel tricks."
				  project_url="/Machine-Learning/projects/support-vector-machine.html"
				  notebook_url="/Machine-Learning/projects/support-vector-machine/notebook"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/10.%20Support%20Vector%20Machine/support-vector-machine.ipynb"
				%}

				{% include project_card.html
				  title="Project 11: Random Forest Classifier"
				  tag="Classification"
				  category="classification"
				  difficulty="intermediate"
				  status="complete"
				  description="Ensemble learning with bagging and feature sampling."
				  project_url="/Machine-Learning/projects/random-forest.html"
				  notebook_url="/Machine-Learning/projects/random-forest/notebook"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/11.%20Random%20Forest%20Classifier/random-forest.ipynb"
				%}

				{% include project_card.html
				  title="Project 12: K-Fold Cross Validation"
				  tag="Evaluation"
				  category="evaluation"
				  difficulty="beginner"
				  status="complete"
				  description="Robust performance estimation across folds."
				  project_url="/Machine-Learning/projects/k-fold-cross-validation.html"
				  notebook_url="/Machine-Learning/projects/k-fold-cross-validation/notebook"
				  colab_url="https://colab.research.google.com/github/sanjanb/Machine-Learning/blob/main/projects/12.%20K-Fold%20Cross%20Validation/k-fold-cross-validation.ipynb"
				%}
1. Choose a project that matches your current learning level
2. Review the related concept pages first
3. Follow the implementation step-by-step
4. Experiment with the code and parameters
5. Document your findings and insights

_New projects are added regularly as we progress through the course!_
