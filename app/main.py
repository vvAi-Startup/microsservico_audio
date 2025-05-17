from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routes.denoise_router import router as audio_router

app = FastAPI(
    title="Microsserviço de Supressão de Ruído",
    description="API que recebe arquivos de áudio e retorna o áudio limpo com base no modelo treinado.",
    version="1.0.0"
)

# Configurar CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, especifique as origens permitidas
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Incluir rotas
app.include_router(audio_router, prefix="/audio")

@app.get("/")
def read_root():
    return {
        "mensagem": "Microsserviço de limpeza de áudio iniciado com sucesso",
        "documentação": "/docs",
        "versão": "1.0.0"
    }
