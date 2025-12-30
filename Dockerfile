FROM python:3.12-slim
WORKDIR /app

# 1. Install Git (Required for your subprocess calls)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# 2. Initialize Git and DVC inside the container so the commands work
RUN git init && dvc init --no-scm -f
EXPOSE 8000
CMD ["python", "backend.py"]