import pickle
import re

# Load the trained model
with open("model/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

# Load the TF-IDF vectorizer
with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# Get message from user
message = input("\nEnter your message: ")

# Clean the message
cleaned_message = clean_text(message)

# Convert message into TF-IDF features
message_vector = vectorizer.transform([cleaned_message])

# Predict
prediction = model.predict(message_vector)

# Display result
if prediction[0] == 1:
    print("\nResult: SPAM")
else:
    print("\nResult: HAM")