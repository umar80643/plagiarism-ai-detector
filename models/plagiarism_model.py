import pandas as pd
from utils.preprocess import clean_text
from utils.similarity import calculate_cosine_similarity

def detect_plagiarism(user_text):
    df = pd.read_csv("datasets/plagiarism.csv")
    user_text = clean_text(user_text)
    max_similarity = 0

    for text in df['text']:
        text = clean_text(str(text))
        similarity = calculate_cosine_similarity(user_text, text)
        if similarity > max_similarity:
            max_similarity = similarity
    return round(max_similarity, 2)