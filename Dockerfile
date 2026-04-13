FROM python:3.14-slim

WORKDIR /app

COPY requirements.txt .
RUN python -m pip install --no-cache-dir -r requirements.txt

COPY . .

# Generate database and model if not already present
RUN python init_db.py

EXPOSE 5000

ENV FLASK_DEBUG=0
ENV FLASK_SECRET_KEY=super-secret-key
ENV APP_HOST=0.0.0.0
ENV APP_PORT=5000

CMD ["python", "app.py"]
