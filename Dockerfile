FROM python:3.9-slim

WORKDIR /app

# Instalar apenas as dependências essenciais
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copiar apenas o arquivo de requisitos primeiro
COPY requirements.txt .

# Instalar dependências Python e limpar cache
RUN pip install --no-cache-dir -r requirements.txt \
    && rm -rf /root/.cache/pip/*

# Copiar apenas os arquivos necessários
COPY app/ app/
COPY .env .

# Expor a porta
EXPOSE 8000

# Comando para iniciar a aplicação
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"] 