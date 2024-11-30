from flask import Flask, request, jsonify, render_template
import pandas as pd
import numpy as np
import pickle
from sklearn.metrics.pairwise import cosine_similarity
import re

# Ініціалізація Flask
app = Flask(__name__, static_folder='static', template_folder='templates')

# Завантаження моделі, даних і векторизатора
with open('book_recommender.pkl', 'rb') as model_file:
    tfidf_matrix, data, vectorizer = pickle.load(model_file)

# 1. Функція для отримання схожих книг
def recommend_books(book_title, top_n=5):
    # Завантаження моделі
    with open("book_recommender.pkl", "rb") as model_file:
        tfidf_matrix, data, vectorizer = pickle.load(model_file)

    # Знаходження індексу книги за її назвою
    matching_books = data[data['Book'].str.lower() == book_title.lower()]
    if matching_books.empty:
        raise ValueError(f"Book '{book_title}' not found in the dataset.")

    book_idx = matching_books.index[0]  # Якщо книга знайдена, беремо перший індекс

    # Знаходження схожості
    cosine_sim = cosine_similarity(tfidf_matrix[book_idx], tfidf_matrix).flatten()
    similar_indices = cosine_sim.argsort()[-top_n - 1:-1][::-1]

    # Результати
    recommendations = data.iloc[similar_indices][['Book', 'Author', 'Genres', 'Avg_Rating']]
    return recommendations


# 2. Маршрут для пошуку книг
@app.route('/books', methods=['GET'])
def get_books():
    search_query = request.args.get('query', '').lower()
    if search_query:
        try:
            search_query = re.escape(search_query)
            filtered_books = data[data['Book'].str.lower().str.contains(search_query, na=False)]
        except re.error as e:
            return jsonify({"error": f"Invalid query format: {str(e)}"}), 400
    else:
        filtered_books = data

    return jsonify(filtered_books[['Book', 'Author', 'Genres']].to_dict(orient='records'))

# 3. Маршрут для рекомендацій книг
@app.route('/recommend', methods=['POST'])
def recommend():
    selected_books = request.json.get('read_books', [])
    if not selected_books:
        return jsonify({"error": "Please select at least one book"}), 400

    read_books_titles = []  # Місце для збереження назв книг
    for book in selected_books:
        book_title = book.get('title', '')

        matched_books = data[data['Book'].str.lower() == book_title.lower()]
        if matched_books.empty:
            return jsonify({"error": f"Book '{book_title}' not found"}), 400

        read_books_titles.append(book_title)

    # Зібрати рекомендації для кожної з книг
    recommendations = pd.DataFrame()
    for book_title in read_books_titles:
        try:
            book_recommendations = recommend_books(book_title)
            recommendations = pd.concat([recommendations, book_recommendations])
        except ValueError as e:
            return jsonify({"error": f"Error in recommendation generation: {str(e)}"}), 500

    recommendations = recommendations.drop_duplicates(subset=['Book'])  # Уникнути повторів
    recommendations_dict = recommendations.to_dict(orient='records')  # Перетворюємо на список словників
    return jsonify(recommendations_dict)


# 4. Головна сторінка
@app.route('/')
def index():
    return render_template('index.html')

# 5. Запуск серверу
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
