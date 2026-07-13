"""
Titanic Data Science Project
BCS 404: Introduction to Data Science with Python

This script performs data acquisition, data cleaning, visualization,
statistical analysis, and logistic regression modelling on the Titanic dataset.
"""

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import set_config
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report


set_config(display="text")

BASE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = BASE_DIR / "data" / "train.csv"
IMAGE_DIR = BASE_DIR / "images"
IMAGE_DIR.mkdir(exist_ok=True)


def load_data():
    return pd.read_csv(DATA_PATH)


def clean_data(df):
    cleaned = df.copy()
    cleaned["Age"] = cleaned["Age"].fillna(cleaned["Age"].median())
    cleaned["Embarked"] = cleaned["Embarked"].fillna(cleaned["Embarked"].mode()[0])
    cleaned = cleaned.drop(columns=["Cabin"])
    cleaned = cleaned.drop_duplicates()
    return cleaned


def create_visualizations(df):
    sns.set_theme(style="whitegrid")

    plt.figure(figsize=(8, 5))
    sns.histplot(df["Age"], bins=30, kde=True, color="skyblue", edgecolor="black")
    plt.title("Histogram of Passenger Ages")
    plt.xlabel("Age")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "age_histogram.png", dpi=180)
    plt.close()

    plt.figure(figsize=(7, 5))
    sns.countplot(data=df, x="Pclass", hue="Pclass", order=sorted(df["Pclass"].unique()), palette=["#4C72B0", "#55A868", "#C44E52"], legend=False)
    plt.title("Passenger Class Distribution")
    plt.xlabel("Passenger Class")
    plt.ylabel("Number of Passengers")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "class_distribution.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df, x="Pclass", y="Age", hue="Pclass", palette=["#8172B2", "#64B5CD", "#CCB974"], legend=False)
    plt.title("Age Distribution by Passenger Class")
    plt.xlabel("Passenger Class")
    plt.ylabel("Age")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "age_by_class_boxplot.png", dpi=180)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(data=df, x="Age", y="Fare", hue="Survived", palette={0: "royalblue", 1: "darkorange"}, alpha=0.75)
    plt.title("Scatter Plot of Age versus Fare")
    plt.xlabel("Age")
    plt.ylabel("Fare")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "age_vs_fare_scatter.png", dpi=180)
    plt.close()

    corr = df[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr()
    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Correlation Heatmap of Numerical Variables")
    plt.tight_layout()
    plt.savefig(IMAGE_DIR / "correlation_heatmap.png", dpi=180)
    plt.close()

    pair = sns.pairplot(df[["Survived", "Pclass", "Age", "Fare"]], hue="Survived", palette={0: "teal", 1: "tomato"}, diag_kind="hist")
    pair.fig.suptitle("Pairplot of Selected Numerical Variables", y=1.02)
    pair.savefig(IMAGE_DIR / "pairplot_selected_variables.png", dpi=180, bbox_inches="tight")
    plt.close("all")


def train_model(df):
    features = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X = df[features]
    y = df["Survived"]
    X = pd.get_dummies(X, columns=["Sex", "Embarked"], drop_first=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    _ = model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    print("Accuracy:", accuracy_score(y_test, predictions))
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, predictions))
    print("\nClassification Report:\n", classification_report(y_test, predictions))


def main():
    df = load_data()
    print("Dataset dimensions:", df.shape)
    print("Column names:", list(df.columns))
    print("\nFirst five observations:\n", df.head())
    print("\nData types:\n", df.dtypes)
    print("\nMissing values before cleaning:\n", df.isnull().sum())

    cleaned = clean_data(df)
    print("\nMissing values after cleaning:\n", cleaned.isnull().sum())
    print("\nDuplicates:", df.duplicated().sum())
    print("\nDescriptive statistics:\n", cleaned.describe())
    print("\nFrequency distribution - Survived:\n", cleaned["Survived"].value_counts())
    print("\nCorrelation matrix:\n", cleaned[["Survived", "Pclass", "Age", "SibSp", "Parch", "Fare"]].corr())

    create_visualizations(cleaned)
    train_model(cleaned)


if __name__ == "__main__":
    main()
