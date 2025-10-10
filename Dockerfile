# =====================================================================
# Estágio 1: Builder - Compila o TA-Lib e instala as dependências
# =====================================================================
FROM python:3.12-slim AS builder

# Instala dependências de build do sistema
RUN apt-get update && apt-get install -y \
    build-essential \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Compila o TA-Lib a partir do código-fonte
RUN wget http://prdownloads.sourceforge.net/ta-lib/ta-lib-0.4.0-src.tar.gz && \
    tar -xzf ta-lib-0.4.0-src.tar.gz && \
    cd ta-lib && \
    ./configure --prefix=/usr/local && \
    make && \
    make install

# Instala as dependências Python em um diretório local
COPY pyproject.toml poetry.lock ./
RUN pip install poetry && \
    poetry config virtualenvs.create false && \
    poetry install --no-root

# =====================================================================
# Estágio 2: Final - A imagem de produção, leve e limpa
# =====================================================================
FROM python:3.12-slim

# Instala apenas as dependências de runtime necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 procps && rm -rf /var/lib/apt/lists/*

# Copia as bibliotecas TA-Lib e Python do estágio de build
COPY --from=builder /usr/local/lib/libta_lib.so* /usr/local/lib/
COPY --from=builder /usr/local/lib/python3.12/site-packages /usr/local/lib/python3.12/site-packages

# Define o diretório de trabalho
WORKDIR /app

# Copy application code
COPY . .

# Expose application port (adjust based on your app's needs)
EXPOSE 8001

# Comando para rodar a aplicação (ajuste conforme seu ponto de entrada)
CMD ["python", "src/binance_client.py"]
