# Machine Learning Projects

This repository documents my practical exploration of machine learning, with a particular focus on **model generalization, robustness, and statistical learning**.

As part of my studies in Data Science and Artificial Intelligence at École Polytechnique, I aim to bridge theoretical concepts with empirical experimentation, developing a deeper understanding of how machine learning algorithms behave and generalize to unseen data.

## Current Project: Forest Cover Type Classification

**Status:** In progress — Exploratory Data Analysis

### Overview

This project investigates the classification of forest cover types using cartographic and environmental features, based on the Kaggle *Forest Cover Type Prediction* dataset.

Beyond achieving high predictive accuracy, the objective is to understand **how model complexity, feature selection, and data characteristics affect out-of-sample performance**.

### Research Questions

- How does model complexity influence overfitting and generalization?
- How do different classification algorithms compare on unseen observations?
- Which features contribute most to predictive performance?
- How robust are models to changes in the input data distribution?
- How does the amount of training data affect generalization?

### Methodology

The project is organized into successive stages:

1. **Exploratory Data Analysis:** Examine feature distributions, class balance, correlations, and potentially informative predictors.
2. **Data Preprocessing:** Investigate appropriate feature transformations and preprocessing strategies while avoiding data leakage.
3. **Baseline Model Comparison:** Compare interpretable baseline classifiers with more flexible models, including decision trees and Random Forests.
4. **Generalization Analysis:** Investigate model complexity, overfitting, learning curves, and out-of-sample performance.
5. **Robustness Experiments:** Evaluate model sensitivity to input perturbations and potential distribution shifts.

### Current Progress

The initial exploratory analysis examines:

- Dataset structure and feature types.
- Distribution of the seven forest cover classes.
- Pairwise correlations and potential multicollinearity.
- Relationships between environmental features and forest cover types.

**[View the Exploratory Data Analysis Notebook](notebooks/01_exploratory_analysis.ipynb)**

### Tools

Python, pandas, NumPy, seaborn, scikit-learn, and Jupyter Notebook.

### Dataset

Kaggle — [Forest Cover Type Prediction](https://www.kaggle.com/c/forest-cover-type-prediction/data)

The dataset contains cartographic features such as elevation, slope, soil type, wilderness area, and distances to hydrological features.

The raw CSV files are not included in this repository. They can be downloaded from Kaggle and placed in the project's `data/` directory.

---

*This repository is a work in progress. Additional experiments, models, and findings will be added as the project develops.*

