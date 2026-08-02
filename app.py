import streamlit as st
import joblib
import re
import nltk

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Download NLTK data (runs only once)
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("omw-1.4")

# Load saved model and vectorizer
model = joblib.load("sentiment_model.pkl")
tfidf = joblib.load("tfidf_vectorizer.pkl")

lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z ]', '', text)

    words = word_tokenize(text)
    words = [w for w in words if w not in stopwords.words("english")]
    words = [lemmatizer.lemmatize(w) for w in words]

    return " ".join(words)

st.set_page_config(page_title="Sentiment Analysis", page_icon="😊")

st.title("Women's Clothing Sentiment Analysis")

review = st.text_area("Enter your review")

if st.button("Predict"):

    clean = preprocess(review)

    vector = tfidf.transform([clean])

    prediction = model.predict(vector)

    if prediction[0] == "0" or prediction[0] == 0:
        st.success("😊 Positive Review")
    else:
        st.error("😞 Negative Review")
