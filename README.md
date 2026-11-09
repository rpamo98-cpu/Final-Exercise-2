# Wine Quality Prediction System

## Problem
A small wine-production business needs a consistent preliminary way to screen wine samples using physicochemical measurements.

## Dataset
UCI Wine Quality Dataset — Red Wine CSV:
https://archive.ics.uci.edu/dataset/186/wine+quality

## Prediction rule
- Quality score >= 6: GOOD QUALITY (class 1)
- Quality score < 6: NEEDS IMPROVEMENT (class 0)

The original `quality` column is used to create the target and is excluded from model input features.

## Technologies
Python, pandas, scikit-learn, Streamlit, joblib, GitHub.

## Installation and running
1. Install Python 3.10 or later.
2. Install dependencies: `pip install -r requirements.txt`
3. Train the model and download the dataset: `python train_model.py`
4. Start the application: `streamlit run app.py`
5. Open the local address displayed in the terminal (normally http://localhost:8501).

## Features
- Displays dataset records, average quality score, and good-quality record count.
- Shows a dataset preview.
- Provides a sample input CSV download.
- Accepts a CSV upload and checks required predictor columns.
- Displays predictions and allows downloading prediction results.

## Limitation
This classroom prototype uses a particular red-wine dataset. It may not generalize to other wine types, producers, regions, or testing conditions. Predictions are preliminary screening results, not replacements for laboratory testing, sensory evaluation, food-safety procedures, or professional quality-control decisions.
