import pandas as pd
from utils.preprocess import clean_text
from utils.similarity import calculate_cosine_similarity

def detect_plagiarism(input_text):
    df = pd.read_csv("datasets/plagiarism.csv")
    input_text = clean_text(input_text)
    max_similarity = 0
    for text in df['text']:
        text = clean_text(str(text))
        similarity = calculate_cosine_similarity(input_text, text)
        if similarity > max_similarity:
            max_similarity = similarity
    return round(max_similarity, 2)