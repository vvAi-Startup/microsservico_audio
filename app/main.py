from fastapi import FastAPI
from app.routes.denoise_router import router as audio_router

app = FastAPI(
    title="Microsserviço de Supressão de Ruído",
    description="API que recebe chunks de áudio e retorna o áudio limpo com base no modelo treinado.",
    version="1.0.0"
)

app.include_router(audio_router, prefix="/audio")


@app.get("/")
def read_root():
    return {"mensagem": "Microsserviço de limpeza de áudio iniciado com sucesso"}




