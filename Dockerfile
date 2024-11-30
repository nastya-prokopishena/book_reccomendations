# Використовуємо Python 3.11 образ
FROM python:3.11-slim

# Встановлюємо робочу директорію всередині контейнера
WORKDIR /app

# Копіюємо файли в контейнер
COPY . /app

# Встановлюємо необхідні бібліотеки
RUN pip install --no-cache-dir -r requirements.txt

# Відкриваємо порт, на якому буде працювати сервер
EXPOSE 5000

# Запускаємо Flask сервер
CMD ["python", "app.py"]
