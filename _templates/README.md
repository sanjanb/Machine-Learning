# How to Add New Content

## Adding a New Lecture

1. Copy `_templates/lecture_template.md`
2. Rename to `pages/lectures/lectureX.md` (where X is the lecture number)
3. Fill in the template with your content
4. Add the lecture link to `_data/sidebars/sidebar.yml`
5. Update the navigation in `_layouts/default.html`

## Adding a New Topic

1. Copy `_templates/topic_template.md`
2. Rename to `topics/[topic_name].md`
3. Fill in the template with your content
4. Add the topic link to `_data/sidebars/sidebar.yml`
5. Update the navigation in `_layouts/default.html`

## Quick Commands

### Start Development Server
```bash
cd 'd:\Projects\Teaching\Machine-Learning'
jekyll serve --host 127.0.0.1 --port 4000
```

### Build Site
```bash
cd 'd:\Projects\Teaching\Machine-Learning'
jekyll build
```

## File Structure
```
Machine-Learning/
├── _layouts/           # HTML layouts
├── _data/             # YAML data files
├── _templates/        # Content templates
├── pages/
│   └── lectures/     # Lecture files
├── topics/           # Topic deep-dives
├── intro/            # Course introduction
├── references/       # Books, papers, links
├── Foundations/      # Foundation resources
└── _config.yml       # Site configuration
```

## Writing Tips

### Math Equations
Use LaTeX syntax with `$$` for block equations:
```markdown
$$
\theta = \theta - \alpha \frac{\partial J(\theta)}{\partial \theta}
$$
```

Use `$` for inline equations: `$\theta_0$`

### Code Blocks
```python
# Python code
def gradient_descent(X, y, theta, alpha, iterations):
    m = len(y)
    for i in range(iterations):
        prediction = X.dot(theta)
        theta = theta - (alpha/m) * X.T.dot(prediction - y)
    return theta
```

### Links
- Internal: `[Link text](/path/to/page.html)`
- External: `[Link text](https://external-url.com)`