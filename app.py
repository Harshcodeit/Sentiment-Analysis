import streamlit as st
import pandas as pd
import joblib
import string
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk

try:
    nltk.data.find('tokenizers/punkt')
except:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except:
    nltk.download('stopwords')

stop_words=set(stopwords.words('english'))

def clean_text(txt):
    # make string lower case
    txt=txt.lower()

    # remove punctuations
    txt=txt.translate(str.maketrans('','',string.punctuation))

    # remove numbers and emoji
    new=""
    for i in txt:
        if i.isascii() and not i.isdigit():
            new+=i

    # remove stop words
    words=word_tokenize(new)
    cleaned=[]
    for i in words:
        if not i in stop_words:
            cleaned.append(i)

    return ' '.join(cleaned)


emotion_map = {
    0: ("😢 Sadness", st.error),
    1: ("😠 Anger", st.error),
    2: ("❤️ Love", st.success),
    3: ("😲 Surprise", st.info),
    4: ("😨 Fear", st.warning),
    5: ("😊 Joy", st.success)
}



model=joblib.load('logistic_regression_model.pkl')
vectorizer=joblib.load('tfidf_vectorizer.pkl')

# emotions=['sadness','anger','love','surprise','fear','joy']

st.set_page_config(
    page_title="Emotion Detector",
    page_icon="😊",
    layout='centered'
)

st.title("😊 Emotion Detection App")
st.markdown("Enter a sentence and I'll predict the emotion behind it.")

txt=st.text_area(
    "Enter Text",
    placeholder="Example: I am enjoying the movie today!",
    height=120
)

# When predict is clicked
if st.button("🔍 Analyze Emotion"):
    if txt.strip() == "":
        st.warning("Please enter some text.")
        st.stop()

    raw_input={
        'text':txt
    }

    # pre-process text
    cleaned_txt=clean_text(raw_input['text'])

    with st.spinner("Analyzing Emotion..."):
        # vectorize
        X=vectorizer.transform([cleaned_txt])

        # predict sentiment
        pred = model.predict(X)[0]

    # show result
    emotion, display=emotion_map[pred]
    st.subheader("Prediction")

    display(emotion)

    st.markdown("---")
    st.caption("Built with Streamlit, TF-IDF and Logistic Regression")
    

