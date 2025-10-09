# Use a lightweight Python image
FROM python:3.12-slim AS builder

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy TA-Lib source
COPY ta-lib-0.4.0-src.tar.gz .

# Build TA-Lib from source
RUN tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install

# Install Python dependencies
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# Copy application code
COPY . .

# Expose application port (adjust based on your app's needs)
EXPOSE 8001

# Command to run the application (adjust based on your entry point)
CMD ["python", "src/binance_client.py"]
