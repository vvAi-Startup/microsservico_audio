from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
import os
import tempfile
from typing import Optional
from services.audio_service import AudioService

router = APIRouter()
audio_service = AudioService()

@router.post("/denoise")
async def denoise_audio(
    file: UploadFile = File(...),
    intensity: Optional[float] = 1.0,
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
    try:
        # Validar intensidade
        if not 0.0 <= intensity <= 1.0:
            raise HTTPException(status_code=400, detail="Intensidade deve estar entre 0.0 e 1.0")
        
        # Criar arquivo temporário para o áudio de entrada
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()
            
            # Processar áudio
            output_file, process_dir = await audio_service.process_audio(temp_file.name, intensity)
            
            # Configurar limpeza do arquivo temporário
            if background_tasks:
                background_tasks.add_task(os.unlink, temp_file.name)
            
            # Retornar arquivo processado
            return FileResponse(
                output_file,
                media_type="audio/wav",
                filename=f"denoised_{file.filename}"
            )
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

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

