FROM python:3.10-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy app code
COPY . .

# Ensure SQLite directory exists
RUN mkdir -p /app/data && chmod 777 /app/data

EXPOSE 8000

CMD ["python", "main.py"]
