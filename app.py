import streamlit as st
import pickle
import re

# -----------------------------
# Load Model and Vectorizer
# -----------------------------

with open("model/spam_model.pkl", "rb") as file:
    model = pickle.load(file)

with open("model/tfidf_vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)


# -----------------------------
# Text Cleaning
# -----------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# -----------------------------
# Page Configuration
# -----------------------------

st.set_page_config(
    page_title="Email Spam Classifier",
    page_icon="📧",
    layout="centered"
)


# -----------------------------
# Custom CSS
# -----------------------------

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 38px;
    font-weight: bold;
}

.subtitle {
    text-align: center;
    font-size: 18px;
    margin-bottom: 30px;
}

</style>
""", unsafe_allow_html=True)


# -----------------------------
# Header
# -----------------------------

st.markdown(
    '<div class="main-title">📧 Email Spam Classifier</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">Detect whether a message is Spam or Ham using Machine Learning</div>',
    unsafe_allow_html=True
)


# -----------------------------
# Message Input
# -----------------------------

message = st.text_area(
    "Enter your message",
    height=180,
    placeholder="Type or paste your email/message here..."
)


# -----------------------------
# Prediction
# -----------------------------

if st.button("🔍 Check Message", use_container_width=True):

    if not message.strip():

        st.warning("⚠️ Please enter a message first.")

    else:

        cleaned_message = clean_text(message)

        message_vector = vectorizer.transform(
            [cleaned_message]
        )

        prediction = model.predict(message_vector)

        # Probability
        probability = model.predict_proba(message_vector)

        spam_probability = probability[0][1]
        ham_probability = probability[0][0]

        # -----------------------------
        # Result
        # -----------------------------

        if prediction[0] == 1:

            st.error("🚨 SPAM MESSAGE")

            st.write(
                f"Spam Probability: **{spam_probability * 100:.2f}%**"
            )

        else:

            st.success("✅ HAM MESSAGE")

            st.write(
                f"Ham Probability: **{ham_probability * 100:.2f}%**"
            )

        # -----------------------------
        # Probability Chart
        # -----------------------------

        st.subheader("Prediction Probability")

        chart_data = {
            "Ham": ham_probability,
            "Spam": spam_probability
        }

        st.bar_chart(chart_data)