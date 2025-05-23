from fastapi import APIRouter, UploadFile, File, Form, HTTPException, BackgroundTasks, Request, Query
from fastapi.responses import FileResponse
import os
import tempfile
from typing import Optional
from app.services.denoiser_service import AudioService


router = APIRouter()
audio_service = AudioService()

@router.post("/denoise")
async def process_audio(
    request: Request,
    audio_file: UploadFile = File(...),
    intensity: Optional[float] = Query(1.0),
    session_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None),
    filename: Optional[str] = Query(None),
    background_tasks: BackgroundTasks = None
):
    """
    Endpoint para processar um arquivo de áudio e remover ruído.
    
    Args:
        audio_file: Arquivo de áudio a ser processado
        intensity: Intensidade do processamento (0.0 a 1.0)
        session_id: ID da sessão (opcional)
        user_id: ID do usuário (opcional)
        filename: Nome do arquivo (opcional)
        background_tasks: Tarefas em segundo plano do FastAPI
    """
    try:
        print("\n=== DEBUG DA REQUISIÇÃO ===")
        print(f"URL: {request.url}")
        print(f"Headers: {dict(request.headers)}")
        print(f"Query Params: {dict(request.query_params)}")
        print(f"Content-Type do arquivo: {audio_file.content_type}")
        print(f"Nome do arquivo: {audio_file.filename}")
        print(f"Intensidade: {intensity}")
        print(f"Session ID: {session_id}")
        print(f"User ID: {user_id}")
        print(f"Filename: {filename}")
        
        # Validar intensidade
        try:
            intensity = float(intensity)
            if not 0.0 <= intensity <= 1.0:
                print(f"Erro: Intensidade inválida: {intensity}")
                raise HTTPException(status_code=400, detail="Intensidade deve estar entre 0.0 e 1.0")
        except (ValueError, TypeError):
            print(f"Erro: Intensidade inválida: {intensity}")
            raise HTTPException(status_code=400, detail="Intensidade deve ser um número entre 0.0 e 1.0")
        
        # Verificar se é um arquivo de áudio
        if not audio_file.content_type or not audio_file.content_type.startswith('audio/'):
            print(f"Erro: Tipo de arquivo inválido: {audio_file.content_type}")
            raise HTTPException(status_code=400, detail="Arquivo deve ser do tipo audio/*")

        # Criar arquivo temporário para o áudio de entrada
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await audio_file.read()
            print(f"Tamanho do arquivo recebido: {len(content)} bytes")
            temp_file.write(content)
            temp_file.flush()
            print(f"Arquivo temporário criado: {temp_file.name}")

            # Processar áudio
            output_file, process_dir = await audio_service.process_audio(temp_file.name, intensity)
            print(f"Áudio processado: {output_file}")
            
            # Configurar limpeza do arquivo temporário
            if background_tasks:
                background_tasks.add_task(os.unlink, temp_file.name)
        
            # Preparar headers
            headers = {
                "Content-Disposition": f"attachment; filename=denoised_{audio_file.filename}"
            }
            
            # Adicionar headers opcionais se fornecidos
            if session_id:
                headers["X-Session-ID"] = session_id
            if user_id:
                headers["X-User-ID"] = user_id
            if filename:
                headers["X-Original-Filename"] = filename

            print("=== FIM DO DEBUG (SUCESSO) ===\n")
            return FileResponse(
                output_file,
                media_type="audio/wav",
                filename=f"denoised_{audio_file.filename}",
                headers=headers
            )

    except Exception as e:
        print("\n=== ERRO NA REQUISIÇÃO ===")
        print(f"Tipo do erro: {type(e).__name__}")
        print(f"Mensagem do erro: {str(e)}")
        import traceback
        print("Stack trace completo:")
        traceback.print_exc()
        print("=== FIM DO ERRO ===\n")
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