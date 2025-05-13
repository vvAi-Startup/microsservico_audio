from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"mensagem": "Microsserviço de limpeza de áudio iniciado com sucesso"}




