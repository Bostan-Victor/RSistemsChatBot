FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["gunicorn", "wsgi:app", "--workers", "1", "--threads", "8", "--timeout", "60", "--bind", "0.0.0.0:8080"]
