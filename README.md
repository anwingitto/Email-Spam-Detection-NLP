# Spam Email Classification Using NLP

## 1. Project Overview

This project is a Natural Language Processing (NLP) based system for
classifying emails as either Spam or Ham.

The project uses TF-IDF for converting email text into numerical features
and Multinomial Naive Bayes for classification.

The main objective is to automatically identify unwanted or suspicious
emails and separate them from normal emails.

---

## 2. Problem Statement

Spam emails are unwanted messages that may contain advertisements,
fraudulent content, or suspicious links.

Manually identifying spam emails can be difficult when the number of
emails is large.

This project develops an NLP-based machine learning system that can
automatically classify email messages as:

- Spam
- Ham (Not Spam)

---

## 3. Objectives

The main objectives of this project are:

- To understand the use of NLP for text classification.
- To preprocess email text data.
- To convert text into numerical features using TF-IDF.
- To train a machine learning model for spam classification.
- To evaluate the performance of the model.
- To classify new email messages as Spam or Ham.

---

## 4. Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Natural Language Processing (NLP)

---

## 5. Machine Learning Model

The project uses:

### TF-IDF

TF-IDF (Term Frequency-Inverse Document Frequency) is used to convert
text data into numerical feature vectors.

It gives importance to words based on their frequency in the messages
and their occurrence across the dataset.

### Multinomial Naive Bayes

Multinomial Naive Bayes is used as the classification algorithm.

It is suitable for text classification problems such as spam email
detection.

---

## 6. Dataset

The dataset contains email/message text and its corresponding class label.

The labels are:

- 0 = Ham
- 1 = Spam

The dataset is stored inside the `dataset` folder.

---

## 7. Text Preprocessing

The following preprocessing steps are performed:

1. Convert text to lowercase.
2. Remove URLs.
3. Remove email addresses.
4. Remove punctuation and special characters.
5. Remove extra spaces.

After preprocessing, the cleaned text is used for feature extraction.

---

## 8. Methodology

The project follows these steps:

```text
Dataset
   ↓
Data Cleaning
   ↓
Text Preprocessing
   ↓
TF-IDF Feature Extraction
   ↓
Train-Test Split
   ↓
Multinomial Naive Bayes
   ↓
Prediction
   ↓
Model Evaluation