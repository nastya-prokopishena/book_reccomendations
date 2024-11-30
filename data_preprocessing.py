import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# 1. Завантаження та попередня обробка даних
data = pd.read_csv("goodreads_data.csv")
data.dropna(subset=["Book", "Author", "Description", "Genres"], inplace=True)

# Об'єднання текстових стовпців
data['combined_text'] = data['Author'] + " " + data['Genres'] + " " + data['Description']

# 2. Векторизація даних
vectorizer = TfidfVectorizer(stop_words='english', max_features=10000)
tfidf_matrix = vectorizer.fit_transform(data['combined_text'])

# 3. Збереження моделі та векторизатора
with open("book_recommender.pkl", "wb") as model_file:
    pickle.dump((tfidf_matrix, data, vectorizer), model_file)

print("Модель збережена успішно.")

# 4. Функція для отримання схожих книг
def recommend_books(book_title, top_n=5):
    # Завантаження моделі
    with open("book_recommender.pkl", "rb") as model_file:
        tfidf_matrix, data, vectorizer = pickle.load(model_file)

    # Знаходження схожості
    book_idx = data[data['Book'] == book_title].index[0]
    cosine_sim = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n-1:-1][::-1]

    # Результати
    recommendations = data.iloc[similar_indices][['Book', 'Author', 'Genres', 'Avg_Rating']]
    return recommendations

