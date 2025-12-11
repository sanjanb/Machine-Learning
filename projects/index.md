---
title: "Projects"
keywords: cs229, machine learning, projects
sidebar: sidebar
permalink: projects/
layout: default
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
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/5.%20Dummy%20variable%20%26%20one-hot%20encoding/homeprices%20(2).csv">Dataset</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/5.%20Dummy%20variable%20%26%20one-hot%20encoding/carprices.csv">Dataset</a>
    	</p>
    </article>

    <!-- Project 6 -->
    <article class="card" data-category="preprocessing" data-difficulty="beginner" data-status="planned">
    	<header>
    		<h3>Project 6: Train/Test Split</h3>
    		<span class="tag">Evaluation</span>
    	</header>
    	<p>Proper dataset splitting and evaluation.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/train-test-split.html">Project page</a>
    		<span>·</span>
    		<span class="muted">Notebook coming soon</span>
    	</p>
    </article>

    <!-- Project 7 -->
    <article class="card" data-category="classification" data-difficulty="beginner" data-status="complete">
    	<header>
    		<h3>Project 7: Logistic Regression</h3>
    		<span class="tag">Classification</span>
    	</header>
    	<p>Binary classification with logistic regression.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/logistic-regression.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/logistic-regression/notebook">Notebook</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/7.%20Logistic%20Regression/insurance_data.csv">Dataset</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/7.%20Logistic%20Regression/HR_comma_sep.csv">Dataset</a>
    	</p>
    </article>

    <!-- Project 8 -->
    <article class="card" data-category="classification" data-difficulty="intermediate" data-status="complete">
    	<header>
    		<h3>Project 8: Multiclass Logistic Regression</h3>
    		<span class="tag">Classification</span>
    	</header>
    	<p>Softmax regression and OvR strategies.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/multiclass-logistic-regression.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/multiclass-logistic-regression/notebook">Notebook</a>
    	</p>
    </article>

    <!-- Project 9 -->
    <article class="card" data-category="classification" data-difficulty="intermediate" data-status="complete">
    	<header>
    		<h3>Project 9: Decision Trees</h3>
    		<span class="tag">Classification</span>
    	</header>
    	<p>Tree-based models for classification/regression.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/decision-trees.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/decision-trees/notebook">Main notebook</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/decision-trees/exercise/notebook">Exercise</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/9.%20Decision%20Trees/salaries.csv">Dataset</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/9.%20Decision%20Trees/Exercise/titanic.csv">Dataset</a>
    	</p>
    </article>

    <!-- Project 10 -->
    <article class="card" data-category="classification" data-difficulty="intermediate" data-status="complete">
    	<header>
    		<h3>Project 10: Support Vector Machine</h3>
    		<span class="tag">Classification</span>
    	</header>
    	<p>Classification with optimal hyperplanes and kernel tricks.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/support-vector-machine.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/support-vector-machine/notebook">Notebook</a>
    	</p>
    </article>

    <!-- Project 11 -->
    <article class="card" data-category="classification" data-difficulty="intermediate" data-status="complete">
    	<header>
    		<h3>Project 11: Random Forest Classifier</h3>
    		<span class="tag">Classification</span>
    	</header>
    	<p>Ensemble learning with bagging and feature sampling.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/random-forest.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/random-forest/notebook">Notebook</a>
    	</p>
    </article>

    <!-- Project 12 -->
    <article class="card" data-category="evaluation" data-difficulty="beginner" data-status="complete">
    	<header>
    		<h3>Project 12: K-Fold Cross Validation</h3>
    		<span class="tag">Evaluation</span>
    	</header>
    	<p>Robust performance estimation across folds.</p>
    	<p class="links">
    		<a class="nav-link" href="/Machine-Learning/projects/k-fold-cross-validation.html">Project page</a>
    		<span>·</span>
    		<a class="nav-link" href="/Machine-Learning/projects/k-fold-cross-validation/notebook">Notebook</a>
    	</p>
    </article>

</section>

<section class="cta">
	<div class="cta-card">
		<h3>Want to contribute?</h3>
		<p>Open an issue or PR with a new dataset, notebook, or improvement.</p>
		<a class="btn" href="https://github.com/sanjanb/Machine-Learning">View on GitHub</a>
	</div>
</section>

<script>
	(function(){
		const grid = document.getElementById('projects-grid');
		const controls = ['filter-category','filter-difficulty','filter-status'].map(id=>document.getElementById(id));
		function applyFilters(){
			const criteria = {
				category: document.getElementById('filter-category').value,
				difficulty: document.getElementById('filter-difficulty').value,
				status: document.getElementById('filter-status').value
			};
			[...grid.querySelectorAll('.card')].forEach(card=>{
				const match = (!criteria.category || criteria.category==='all' || card.dataset.category===criteria.category)
					&& (!criteria.difficulty || criteria.difficulty==='all' || card.dataset.difficulty===criteria.difficulty)
					&& (!criteria.status || criteria.status==='all' || card.dataset.status===criteria.status);
				card.style.display = match ? '' : 'none';
			});
		}
		controls.forEach(el=>el.addEventListener('change', applyFilters));
	})();
</script>

<style>
	:root {
		--bg: #0f172a;
		--panel: #111827;
		--text: #e5e7eb;
		--muted: #9ca3af;
		--primary: #60a5fa;
		--accent: #34d399;
		--border: #1f2937;
	}
	.hero {
		background: linear-gradient(135deg, #0f172a 0%, #1f2937 100%);
		color: var(--text);
		padding: 3rem 1rem;
		border-radius: 12px;
		margin-bottom: 1.5rem;
	}
	.hero-content { max-width: 900px; margin: 0 auto; }
	.hero h1 { margin: 0 0 0.5rem 0; font-size: 2rem; }
	.hero p { margin: 0; color: var(--muted); }
	.hero-actions { margin-top: 1rem; display: flex; gap: 0.5rem; flex-wrap: wrap; }
	.btn { display: inline-block; padding: 0.5rem 0.9rem; border: 1px solid var(--border); border-radius: 8px; color: var(--text); text-decoration: none; background: var(--panel); }
	.btn.primary { border-color: var(--primary); background: #0b1220; color: #cfe8ff; }

	.filters { margin: 1rem 0 0.5rem; }
	.filter-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 0.75rem; }
	.filter { display: grid; gap: 0.35rem; color: var(--muted); }
	select { padding: 0.4rem; border-radius: 6px; border: 1px solid var(--border); background: #0b1220; color: var(--text); }

	.cards { display: grid; grid-template-columns: repeat(auto-fill,minmax(280px,1fr)); gap: 1rem; }
	.card { border: 1px solid var(--border); border-radius: 10px; padding: 1rem; background: #0b1220; color: var(--text); box-shadow: 0 2px 10px rgba(0,0,0,0.15); }
	.card header { display:flex; align-items:center; justify-content: space-between; gap: 0.5rem; }
	.tag { display:inline-block; font-size: 0.75rem; color: #d1fae5; background: #064e3b; padding: 0.15rem 0.5rem; border-radius: 999px; }
	.links { display:flex; flex-wrap:wrap; gap: 0.4rem; align-items:center; }
	.nav-link { color: #cfe8ff; text-decoration: none; }
	.nav-link:hover { text-decoration: underline; }
	.muted { color: var(--muted); }

	.cta { margin-top: 2rem; }
	.cta-card { border: 1px solid var(--border); border-radius: 10px; padding: 1rem; background: #0b1220; color: var(--text); }

	/* Print-friendly */
	@media print { .filters, .hero-actions, .cta { display:none; } }
</style>

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

| Concept             | Theory Page                                                    | Used in Projects |
| ------------------- | -------------------------------------------------------------- | ---------------- |
| Linear Regression   | [📖 Theory](/Machine-Learning/topics/linear_regression.html)   | Project 1        |
| Logistic Regression | [📖 Theory](/Machine-Learning/topics/logistic_regression.html) | Project 2        |
| Gradient Descent    | [📖 Theory](/Machine-Learning/topics/gradient_descent.html)    | Projects 1, 2, 3 |
| Cost Functions      | [📖 Theory](/Machine-Learning/topics/cost_functions.html)      | Projects 1, 2, 3 |

---

## 🚀 Getting Started

1. Choose a project that matches your current learning level
2. Review the related concept pages first
3. Follow the implementation step-by-step
4. Experiment with the code and parameters
5. Document your findings and insights

_New projects are added regularly as we progress through the course!_
