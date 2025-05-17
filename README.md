# Microsserviço de Supressão de Ruído

Este é um microsserviço que utiliza um modelo de aprendizado profundo (U-Net) para remover ruído de arquivos de áudio.

## Requisitos

- Python 3.8+
- PyTorch
- FastAPI
- Outras dependências listadas em `requirements.txt`

## Instalação

1. Clone o repositório:
```bash
git clone <url-do-repositorio>
cd microsservico_audio
```

2. Crie um ambiente virtual e ative-o:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

4. Coloque o modelo treinado em `app/model/best_denoiser_model.pth`

## Executando o Serviço

Para iniciar o servidor:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

O serviço estará disponível em `http://localhost:8000`

## Endpoints

### POST /audio/denoise
Processa um arquivo de áudio para remover ruído.

**Parâmetros:**
- `file`: Arquivo de áudio (WAV, MP3, OGG, FLAC)
- `intensity`: Intensidade do processamento (0.0 a 1.0, opcional, padrão: 1.0)

**Resposta:**
- Arquivo de áudio processado

### GET /audio/health
Verifica o status do serviço.

**Resposta:**
```json
{
    "status": "ok",
    "model_loaded": true,
    "device": "cuda" // ou "cpu"
}
```

## Documentação da API

A documentação interativa da API está disponível em:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Exemplo de Uso

```python
import requests

# URL do serviço
url = "http://localhost:8000/audio/denoise"

# Arquivo de áudio para processar
files = {
    'file': ('audio.wav', open('audio.wav', 'rb'), 'audio/wav')
}

# Parâmetros
params = {
    'intensity': 0.8
}

# Enviar requisição
response = requests.post(url, files=files, params=params)

# Salvar resultado
if response.status_code == 200:
    with open('audio_denoised.wav', 'wb') as f:
        f.write(response.content)
```

## Estrutura do Projeto

```
microsservico_audio/
├── app/
│   ├── model/
│   │   ├── denoiser_model.py
│   │   └── best_denoiser_model.pth
│   ├── routes/
│   │   └── denoise_router.py
│   ├── services/
│   │   └── audio_service.py
│   └── main.py
├── requirements.txt
└── README.md
```

## Notas

- O serviço processa arquivos de áudio em lotes para evitar problemas de memória
- Os arquivos processados são salvos em `~/denoiser_output`
- O modelo deve ser treinado previamente e colocado em `app/model/best_denoiser_model.pth` 