FROM python:3.12-slim
WORKDIR /app

# 1. Install Git and build dependencies
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 2. Initialize Git and DVC 
RUN git init && dvc init --no-scm -f

# 3. Dynamic Port Binding for Google Cloud Run
# Cloud Run injects the $PORT variable (usually 8080)
# We use 0.0.0.0 to allow the Google Load Balancer to reach the container
EXPOSE 8080
CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8080}