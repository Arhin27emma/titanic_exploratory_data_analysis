# Titanic Data Science Project

## Project Title
Exploratory Data Analysis, Statistical Analysis and Machine Learning on the Titanic Dataset Using Python

## Course
BCS 404: Introduction to Data Science with Python

## Overview
This project analyses the Titanic dataset using Python. It covers data acquisition, data cleaning, data visualisation, statistical analysis, and a Logistic Regression machine learning model for predicting passenger survival.

## Project Structure

```text
titanic-data-science-project/
├── data/
│   ├── train.csv
│   └── cleaned_titanic.csv
├── images/
│   ├── age_histogram.png
│   ├── class_distribution.png
│   ├── age_by_class_boxplot.png
│   ├── age_vs_fare_scatter.png
│   ├── correlation_heatmap.png
│   └── pairplot_selected_variables.png
├── notebooks/
│   └── titanic_analysis.ipynb
├── report/
│   ├── titanic_report.pdf
│   ├── titanic_report.docx
│   └── analysis_summary.json
├── src/
│   └── titanic_analysis.py
├── README.md
```

## Dataset
The project uses the Titanic training dataset with 891 rows and 12 columns. The main target variable is `Survived`.

## Main Tasks Completed
- Dataset acquisition and inspection
- Missing value detection and treatment
- Duplicate detection
- Histogram, bar chart, boxplot, scatter plot, heatmap, and pairplot with clearer contrasting colours
- Descriptive statistics and frequency distributions
- Correlation analysis
- Logistic Regression survival prediction model
- Accuracy, confusion matrix, and classification report

## Key Results
- Dataset dimensions: 891 rows and 12 columns
- Duplicate observations found: 0
- Logistic Regression accuracy: 80.45%
- Strongest positive correlation: SibSp and Parch (0.415)
- Strongest negative correlation: Pclass and Fare (-0.549)


```