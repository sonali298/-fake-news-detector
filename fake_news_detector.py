import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import streamlit as st

# Load Data
df_fake = pd.read_csv('Fake.csv')
df_true = pd.read_csv('True.csv')

df_fake['label'] = 0
df_true['label'] = 1
df = pd.concat([df_fake, df_true]).reset_index(drop=True)

# Use only the text and label
df = df[['text', 'label']].dropna()

# Split Data
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.2, random_state=42)

# Vectorization
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Model
model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

# Test accuracy
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)

# Streamlit Web App
st.title("📰 Fake News Detection System")
st.write(f"**Model Accuracy:** {accuracy:.2f}")

user_input = st.text_area("Paste a news article text to check if it's Fake or Real:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("Please enter some text!")
    else:
        user_input_tfidf = vectorizer.transform([user_input])
        prediction = model.predict(user_input_tfidf)[0]
        result = "🟢 Real News" if prediction == 1 else "🔴 Fake News"
        st.success(f"Prediction: {result}")