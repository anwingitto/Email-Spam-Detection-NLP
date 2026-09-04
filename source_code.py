# ============================================================
# SPAM EMAIL CLASSIFICATION USING NLP
# TF-IDF + MULTINOMIAL NAIVE BAYES
# ============================================================

import os
import zipfile
import re

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)


# ============================================================
# 1. DATASET SETUP
# ============================================================

dataset_folder = "dataset"

# Find ZIP files inside the dataset folder
zip_files = [
    file for file in os.listdir(dataset_folder)
    if file.lower().endswith(".zip")
]

if len(zip_files) > 0:

    zip_path = os.path.join(dataset_folder, zip_files[0])

    print("ZIP dataset found:", zip_files[0])
    print("Extracting dataset...")

    extract_folder = os.path.join(dataset_folder, "extracted")

    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_folder)

    print("Dataset extracted successfully!")

else:
    extract_folder = dataset_folder
    print("No ZIP file found. Checking dataset folder directly.")


# ============================================================
# 2. FIND CSV FILE
# ============================================================

csv_file = None

for root, directories, files in os.walk(extract_folder):

    for file in files:

        if file.lower().endswith(".csv"):

            csv_file = os.path.join(root, file)
            break

    if csv_file is not None:
        break


if csv_file is None:

    raise FileNotFoundError(
        "No CSV file was found inside the dataset folder."
    )


print("\nCSV file found:")
print(csv_file)


# ============================================================
# 3. LOAD DATASET
# ============================================================

df = pd.read_csv(csv_file)

print("\nDataset loaded successfully!")

print("Dataset Shape:", df.shape)

print("Column Names:", df.columns.tolist())


# ============================================================
# 4. CHECK DATASET INFORMATION
# ============================================================

print("\nDataset Information:")

df.info()


print("\nMissing Values:")

print(df.isnull().sum())


print("\nDataset Description:")

print(df.describe(include="all"))


# ============================================================
# 5. CHECK CLASS DISTRIBUTION
# ============================================================

print("\nClass Distribution:")

print(df["label"].value_counts())


df["label"].value_counts().plot(kind="bar")

plt.title("Spam vs Ham Distribution")

plt.xlabel("Class")

plt.ylabel("Number of Messages")

plt.show()


# ============================================================
# 6. REMOVE DUPLICATE ROWS
# ============================================================

print("\nNumber of duplicate rows:")

print(df.duplicated().sum())


df = df.drop_duplicates()


print("Dataset shape after removing duplicates:")

print(df.shape)


# ============================================================
# 7. HANDLE MISSING TEXT VALUES
# ============================================================

df = df.dropna(subset=["text"])


print("\nMissing values after cleaning:")

print(df.isnull().sum())


# ============================================================
# 8. TEXT PREPROCESSING
# ============================================================

def clean_text(text):

    # Convert text to string
    text = str(text)

    # Convert text to lowercase
    text = text.lower()

    # Remove URLs
    text = re.sub(
        r"http\S+|www\S+|https\S+",
        "",
        text
    )

    # Remove email addresses
    text = re.sub(
        r"\S+@\S+",
        "",
        text
    )

    # Remove punctuation and special characters
    text = re.sub(
        r"[^a-zA-Z\s]",
        "",
        text
    )

    # Remove extra spaces
    text = re.sub(
        r"\s+",
        " ",
        text
    ).strip()

    return text


# Apply text preprocessing

df["clean_text"] = df["text"].apply(clean_text)


print("\nText preprocessing completed.")


# ============================================================
# 9. PREPARE FEATURES AND TARGET
# ============================================================

# Input feature

X = df["clean_text"]


# Target variable

y = df["label"]


# ============================================================
# 10. TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,

    test_size=0.20,

    random_state=42,

    stratify=y
)


print("\nTraining samples:", len(X_train))

print("Testing samples:", len(X_test))


# ============================================================
# 11. TF-IDF VECTORIZATION
# ============================================================

tfidf = TfidfVectorizer(

    max_features=5000,

    stop_words="english"
)


# Learn vocabulary from training data

X_train_tfidf = tfidf.fit_transform(X_train)


# Transform testing data

X_test_tfidf = tfidf.transform(X_test)


print("\nTraining TF-IDF shape:")

print(X_train_tfidf.shape)


print("Testing TF-IDF shape:")

print(X_test_tfidf.shape)


# ============================================================
# 12. TRAIN MULTINOMIAL NAIVE BAYES MODEL
# ============================================================

model = MultinomialNB()


model.fit(

    X_train_tfidf,

    y_train
)


print("\nModel training completed!")


# ============================================================
# 13. MAKE PREDICTIONS
# ============================================================

y_pred = model.predict(

    X_test_tfidf
)


print("\nFirst 20 predictions:")

print(y_pred[:20])


# ============================================================
# 14. MODEL EVALUATION
# ============================================================

accuracy = accuracy_score(

    y_test,

    y_pred
)


precision = precision_score(

    y_test,

    y_pred,

    pos_label=1
)


recall = recall_score(

    y_test,

    y_pred,

    pos_label=1
)


f1 = f1_score(

    y_test,

    y_pred,

    pos_label=1
)


print("\n======================================")

print("        MODEL EVALUATION")

print("======================================")


print("Accuracy :", accuracy)

print("Precision:", precision)

print("Recall   :", recall)

print("F1-score :", f1)


# ============================================================
# 15. CLASSIFICATION REPORT
# ============================================================

print("\nClassification Report:")


print(

    classification_report(

        y_test,

        y_pred

    )

)


# ============================================================
# 16. CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(

    y_test,

    y_pred
)


print("\nConfusion Matrix:")

print(cm)


plt.figure(figsize=(6, 4))


sns.heatmap(

    cm,

    annot=True,

    fmt="d",

    cmap="Blues"

)


plt.title("Confusion Matrix")

plt.xlabel("Predicted Label")

plt.ylabel("Actual Label")


plt.show()


# ============================================================
# 17. RESULTS GRAPH
# ============================================================

metrics = [

    "Accuracy",

    "Precision",

    "Recall",

    "F1-score"

]


values = [

    accuracy,

    precision,

    recall,

    f1

]


plt.figure(figsize=(8, 5))


plt.bar(

    metrics,

    values

)


plt.title("Model Performance")

plt.xlabel("Metrics")

plt.ylabel("Score")

plt.ylim(0, 1)


plt.show()


# ============================================================
# 18. SAMPLE SPAM / HAM PREDICTIONS
# ============================================================

sample_messages = [

    "Congratulations! You have won a free prize. Click here to claim now.",

    "Hi, can we meet tomorrow to discuss the project?",

    "You have been selected for a cash reward. Claim your money now!",

    "Please send me the assignment document before tomorrow."

]


# Clean sample messages

clean_samples = [

    clean_text(message)

    for message in sample_messages

]


# Convert sample messages into TF-IDF

sample_tfidf = tfidf.transform(

    clean_samples

)


# Predict the class

predictions = model.predict(

    sample_tfidf

)


print("\n======================================")

print("       SAMPLE PREDICTIONS")

print("======================================")


for message, prediction in zip(

    sample_messages,

    predictions

):

    if prediction == 1:

        result = "SPAM"

    else:

        result = "HAM"


    print("\nMessage:", message)

    print("Prediction:", result)

    print("-" * 60)


# ============================================================
# PROJECT COMPLETED
# ============================================================

print("\n======================================")

print("     SPAM EMAIL CLASSIFICATION")

print("          PROJECT COMPLETED")

print("======================================")