
# Use Python 3.11 as our base image
FROM python:3.11-slim

# Set the working directory inside the container
WORKDIR /app

# Configure Python
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Copy API dependencies
COPY requirements-api.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements-api.txt

# Copy our application code
COPY app/ ./app/

# Document the port used by FastAPI
EXPOSE 8000

# Start the FastAPI application
CMD ["uvicorn", "main:app", "--app-dir", "app", "--host", "0.0.0.0", "--port", "8000"]
