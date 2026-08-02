import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download required NLTK resources
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))


def preprocess(text):
    # Lowercase
    text = text.lower()

    # Remove special characters
    text = re.sub(r"[^a-zA-Z\s]", " ", text)

    # Tokenize
    words = word_tokenize(text)

    # Remove stopwords
    words = [word for word in words if word not in stop_words]

    # Lemmatize
    words = [lemmatizer.lemmatize(word) for word in words]

    # Remove short words
    words = [word for word in words if len(word) > 2]

    # Remove numeric words
    words = [word for word in words if not word.isnumeric()]

    # Convert back to sentence
    text = " ".join(words)

    # Remove extra spaces
    text = re.sub(r"\s+", " ", text).strip()

    return text


st.set_page_config(
    page_title="Women's Clothing Sentiment Analysis",
    page_icon="😊",
    layout="centered"
)

st.title("😊 Women's Clothing Sentiment Analysis")

st.write("Enter a customer review below to predict whether it is Positive or Negative.")

review = st.text_area(
    "Enter your review",
    height=150,
    placeholder="Example: This dress is amazing and very comfortable."
)

if st.button("Predict"):

    if review.strip() == "":
        st.warning("Please enter a review.")
    else:

        clean_review = preprocess(review)

        vector = tfidf.transform([clean_review])

        prediction = model.predict(vector)[0]

       

        if prediction == 1:
            st.success("😊 Positive Review")
        else:
            st.error("😞 Negative Review")

        if hasattr(model, "predict_proba"):
            probability = model.predict_proba(vector)

            confidence = probability.max() * 100

            st.write(f"**Confidence:** {confidence:.2f}%")
