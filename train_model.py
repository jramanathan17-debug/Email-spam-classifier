import pandas as pd

# Load the dataset
data = pd.read_csv(
    "dataset/SMSSpamCollection.csv",
    encoding="latin-1",
    usecols=[0, 1]
)

# Rename columns
data.columns = ["label", "message"]

# Display first 5 rows
print(data.head())

# Display dataset size
print("\nDataset Shape:")
print(data.shape)

# Display column names
print("\nColumns:")
print(data.columns)

# Count spam and ham messages
print("\nSpam / Ham Count:")
print(data["label"].value_counts())
import re

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Apply cleaning
data["clean_message"] = data["message"].apply(clean_text)

# Display original and cleaned messages
print("\nOriginal vs Cleaned:")
print(data[["message", "clean_message"]].head())
from sklearn.feature_extraction.text import TfidfVectorizer

# Convert text into numerical features using TF-IDF
vectorizer = TfidfVectorizer()

X = vectorizer.fit_transform(data["clean_message"])

print("\nTF-IDF Shape:")
print(X.shape)
from sklearn.model_selection import train_test_split

# Convert labels into numbers
y = data["label"].map({"ham": 0, "spam": 1})

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:")
print(X_train.shape)

print("\nTesting Data Shape:")
print(X_test.shape)
from sklearn.naive_bayes import MultinomialNB

# Create the Naive Bayes model
model = MultinomialNB()

# Train the model
model.fit(X_train, y_train)

print("\nNaive Bayes model training completed!")
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Make predictions on test data
y_pred = model.predict(X_test)

# Calculate evaluation metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\nModel Evaluation Results:")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)

print("\nConfusion Matrix:")
print(cm)

import pickle

# Save the trained model
with open("model/spam_model.pkl", "wb") as file:
    pickle.dump(model, file)

# Save the TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "wb") as file:
    pickle.dump(vectorizer, file)

print("\nModel and TF-IDF vectorizer saved successfully!")