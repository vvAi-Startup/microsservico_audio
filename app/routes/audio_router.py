from http.client import HTTPException
from fastapi import APIRouter, UploadFile, File, Form
from fastapi.responses import JSONResponse, Response
from app.services.denoiser_service import denoise_audio

router = APIRouter()

@router.post("/process_audio")
async def process_audio(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form(...),
    chunk_number: int = Form(...),
    filename: str = Form(...)
):
    
    try:
        #Lê os bytes do arquivo enviado
        file_bytes = await audio_file.read()
        
        if not filename.endswith(".wav"):
            raise HTTPException(status_code=400, detail="Formato de arquivo inválido. Apenas .wav é suportado.")

        if chunk_number < 0:
            raise HTTPException(status_code=400, detail="chunk_number não pode ser negativo.")

        # Chama o service para processar o áudio
        denoised_bytes = denoise_audio(file_bytes)
        
        # Retorna o novo áudio limpo em .wav
        return Response(content=denoised_bytes, media_type="audio/wav",
                    headers={
                        "X-Session-ID": session_id,
                        "X-User-ID": user_id,
                        "X-Chunk-Number": str(chunk_number),
                        "X-Original-Filename": filename
                    })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao processar áudio: {str(e)}")
