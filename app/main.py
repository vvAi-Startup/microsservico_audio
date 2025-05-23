from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routes.denoise_router import router as audio_router


app = FastAPI(
    title="Microsserviço de Supressão de Ruído",
    description="API que recebe chunks de áudio e retorna o áudio limpo com base no modelo treinado.",
    version="1.0.0"
)

# Configurar CORS para permitir todas as origens
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permite todas as origens
    allow_credentials=True,
    allow_methods=["*"],  # Permite todos os métodos (GET, POST, PUT, DELETE, etc)
    allow_headers=["*"],  # Permite todos os headers
    expose_headers=["*"],  # Expõe todos os headers na resposta
    max_age=3600,  # Cache das configurações CORS por 1 hora
)


app.include_router(audio_router, prefix="/audio")

@app.get("/")
def read_root():
    return {"mensagem": "Microsserviço de limpeza de áudio iniciado com sucesso"}
