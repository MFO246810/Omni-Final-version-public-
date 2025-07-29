FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install gunicorn
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose the port Flask runs on
EXPOSE 5000

# Run the Flask app (replace app.py with your entry point)
CMD ["gunicorn", "app:app", "--workers=2", "--threads=4", "--worker-class=gthread", "--timeout=120", "--bind=127.0.0.1:5000"]