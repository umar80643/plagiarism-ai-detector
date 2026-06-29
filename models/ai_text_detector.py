import joblib
from utils.preprocess import clean_text

model = joblib.load("saved_models/ai_model.pkl")
vectorizer = joblib.load("saved_models/vectorizer.pkl")


def detect_ai_text(user_text):
    user_text = clean_text(user_text)

    text_vector = vectorizer.transform([user_text])

    prediction = model.predict(text_vector)[0]

    probability = model.predict_proba(text_vector).max() * 100

    return round(probability, 2), prediction