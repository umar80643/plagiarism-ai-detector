import pandas as pd
from utils.text_analysis import split_into_sentences
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def calculate_cosine_similarity(text1, text2):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    similarity = cosine_similarity(vectors[0], vectors[1])
    return round(similarity[0][0]*100,2)




def sentence_similarity_analysis(text):

    df = pd.read_csv("datasets/plagiarism.csv")

    dataset = df["text"].tolist()

    sentences = split_into_sentences(text)

    results = []

    for sentence in sentences:

        highest = 0

        for sample in dataset:

            similarity = calculate_cosine_similarity(sentence, sample)

            if similarity > highest:
                highest = similarity

        results.append({
            "sentence": sentence,
            "similarity": highest
        })

    return results