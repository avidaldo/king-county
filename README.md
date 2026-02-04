# House Sales in King County - End-to-End Regression Project

This repository contains a complete end-to-end Machine Learning project for predicting house prices in King County, USA. Dealing with real-world data often requires handling messiness, potential data leakage, and careful validation strategies. This project demonstrates these challenges and their solutions.

## Project Overview

**Goal:** Predict the price of a house based on features like square footage, number of bedrooms, location, etc.
**Type:** Regression.
**Key Learnings:**
- Real-world data cleaning (handling duplicates).
- Preventing **Temporal Data Leakage** in time-dependent datasets.
- Feature Engineering and Preprocessing pipelines.
- Model selection and rigorous evaluation using Train/Validation/Test splits.

## Dataset

The dataset is the famous **King County House Sales** dataset.
- **Source:** Kaggle (downloaded automatically in the notebook).
- **Size:** ~21k observations.
- **Features:** 21 variables (Date, Bedrooms, Bathrooms, Sqft, Floors, Waterfront, View, Condition, Grade, Year Built, etc.).

## Project Workflow

The project is structured into sequential notebooks, each focusing on a critical stage of the ML pipeline:

| Notebook | Description | Key Concepts |
|----------|-------------|--------------|
| `01-eda.ipynb` | **Exploratory Data Analysis**. Initial look at the data structure, distributions, and potential issues. | EDA, Data Loading, Visualization. |
| `02-repeated_ids.ipynb` | **Data Cleaning**. Handling duplicate entries for the same house sold multiple times. | Data Cleaning, Duplicates, Consistency. |
| `03-temporal_leakage.ipynb` | **Splitting Strategy**. Why random splitting is dangerous for time-series-like data. Analysis of price trends over time. | **Temporal Split**, Data Leakage, Train/Test Split. |
| `04a-preprocessing-step-by-step.ipynb` | **Feature Engineering (Step-by-Step)**. Detailed walkthrough of each preprocessing step for learning purposes. | Manual Feature Engineering, Data Leakage Prevention, Fit/Transform Paradigm. |
| `04b-preprocessing-pipeline.ipynb` | **Feature Engineering (Production)**. Complete sklearn pipeline for end-to-end inference. | `sklearn.pipeline`, Custom Transformers, Production Deployment. |
| `05-modeling.ipynb` | **Modeling & Evaluation**. Training multiple models, tuning hyperparameters, and final evaluation. | Baseline, Linear Models (Ridge, Lasso), Ensembles (RandomForest, GradientBoosting), Cross-Validation vs Validation Set. |


## Requirements

The project uses standard Python data science libraries:
- `numpy`, `pandas`
- `matplotlib`, `seaborn`
- `scikit-learn`
- `kagglehub` (for data download)

Ensure you have these installed within your environment (recommended to use `uv` as per course guidelines).
