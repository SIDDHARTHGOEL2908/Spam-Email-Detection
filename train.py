"""
train.py
-----------------------------------------
AI Powered Spam Email Detection
Training Script

Author: Siddharth Goel
"""

import re
import string
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# -----------------------------------------
# Load Dataset
# -----------------------------------------

print("=" * 50)
print("Loading Dataset...")
print("=" * 50)

df = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
df = df.iloc[:, :2]
df.columns = ["label", "message"]

print(df.head())

# -----------------------------------------
# Encode Labels
# -----------------------------------------

df["label"] = df["label"].map({
    "ham": 0,
    "spam": 1
})

# -----------------------------------------
# Text Cleaning
# -----------------------------------------

def clean_text(text):

    text = text.lower()

    text = re.sub(r"http\S+", "", text)

    text = re.sub(r"\d+", "", text)

    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    text = re.sub(r"\s+", " ", text)

    return text.strip()

df["message"] = df["message"].apply(clean_text)

# -----------------------------------------
# Features
# -----------------------------------------

X = df["message"]

y = df["label"]

# -----------------------------------------
# TF-IDF
# -----------------------------------------

print("\nCreating TF-IDF Features...")

vectorizer = TfidfVectorizer(
    stop_words="english",
    max_features=5000
)

X = vectorizer.fit_transform(X)

# -----------------------------------------
# Split Dataset
# -----------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("Training Samples :", X_train.shape[0])
print("Testing Samples  :", X_test.shape[0])

# -----------------------------------------
# Train Model
# -----------------------------------------

print("\nTraining Model...")

model = MultinomialNB()

model.fit(X_train, y_train)

# -----------------------------------------
# Prediction
# -----------------------------------------

predictions = model.predict(X_test)

# -----------------------------------------
# Evaluation
# -----------------------------------------

accuracy = accuracy_score(y_test, predictions)

print("\nAccuracy")
print(f"{accuracy*100:.2f}%")

print("\nClassification Report\n")

print(classification_report(
    y_test,
    predictions
))

print("\nConfusion Matrix\n")

print(confusion_matrix(
    y_test,
    predictions
))

# -----------------------------------------
# Save Files
# -----------------------------------------

joblib.dump(model, "model.pkl")

joblib.dump(vectorizer, "vectorizer.pkl")

print("\nModel Saved Successfully!")

print("model.pkl")

print("vectorizer.pkl")

# -----------------------------------------
# Example Prediction
# -----------------------------------------

sample = [
    "Congratulations! You have won ₹50,000. Click here to claim."
]

sample_vector = vectorizer.transform(sample)

prediction = model.predict(sample_vector)[0]

probability = model.predict_proba(sample_vector).max()

print("\nExample Prediction")

if prediction == 1:
    print("Spam")
else:
    print("Ham")

print("Confidence :", round(probability * 100, 2), "%")
