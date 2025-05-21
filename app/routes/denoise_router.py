from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import tempfile
from typing import Optional
from app.services.denoiser_service import AudioService


router = APIRouter()
audio_service = AudioService()

@router.post("/denoise")
async def process_audio(
    audio_file: UploadFile = File(...),
    intensity: Optional[float] = 1.0,
    session_id: str = Form(...),
    user_id: str = Form(...),
    chunk_number: int = Form(...),
    filename: str = Form(...),
    background_tasks: BackgroundTasks = None
):
    """
    Endpoint para processar um arquivo de áudio e remover ruído.
    
    Args:
        file: Arquivo de áudio a ser processado
        intensity: Intensidade do processamento (0.0 a 1.0)
        background_tasks: Tarefas em segundo plano do FastAPI
        
    Returns:
        FileResponse: Arquivo de áudio processado
    """
    
    print("Campos recebidos:")
    print(f"audio_file: {audio_file}")
    print(f"session_id: {session_id}")
    print(f"user_id: {user_id}")
    print(f"chunk_number: {chunk_number}")
    print(f"filename: {filename}")
    
    try:
        # Validar intensidade
        if not 0.0 <= intensity <= 1.0:
            raise HTTPException(status_code=400, detail="Intensidade deve estar entre 0.0 e 1.0")
        
        if audio_file.content_type != "audio/wav":
            raise HTTPException(status_code=400, detail="Arquivo deve ser do tipo audio/wav")
        
        # Criar arquivo temporário para o áudio de entrada
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(audio_file.filename)[1]) as temp_file:
            content = await audio_file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Processar áudio
            output_file, process_dir = await audio_service.process_audio(temp_file.name, intensity)
            
            # Configurar limpeza do arquivo temporário
            if background_tasks:
                background_tasks.add_task(os.unlink, temp_file.name)
        
            return FileResponse(
                output_file,
                media_type="audio/wav",
                filename=f"denoised_{audio_file.filename}",
                headers={
                "X-Session-ID": session_id,
                            "X-User-ID": user_id,
                            "X-Chunk-Number": str(chunk_number),
                            "X-Original-Filename": filename
            })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento do áudio: {str(e)}")

@router.get("/health")
async def health_check():
    """
    Endpoint para verificar a saúde do serviço.
    
    Returns:
        dict: Status do serviço
    """
    return {
        "status": "ok",
        "model_loaded": audio_service.model is not None,
        "device": str(audio_service.device)
    }