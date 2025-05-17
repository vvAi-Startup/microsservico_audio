from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
import torchaudio
from app.services.denoiser_service import denoise_waveform, SAMPLE_RATE


router = APIRouter()

@router.post("/denoise")
async def process_audio(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form(...),
    chunk_number: int = Form(...),
    filename: str = Form(...)
):
    
    print("Campos recebidos:")
    print(f"audio_file: {audio_file}")
    print(f"session_id: {session_id}")
    print(f"user_id: {user_id}")
    print(f"chunk_number: {chunk_number}")
    print(f"filename: {filename}")
    
    if audio_file.content_type != "audio/wav":
        raise HTTPException(status_code=400, detail="Arquivo deve ser do tipo audio/wav")
    try:
        wav_bytes = await audio_file.read()
        waveform, sr = torchaudio.load(io.BytesIO(wav_bytes))
        
        denoised_audio = denoise_waveform(waveform, sr)
        
        buffer = io.BytesIO()
        torchaudio.save(buffer, denoised_audio.unsqueeze(0), SAMPLE_RATE, format="wav")
        buffer.seek(0)
        
        return StreamingResponse(buffer, media_type="audio/wav", headers={
            "X-Session-ID": session_id,
                        "X-User-ID": user_id,
                        "X-Chunk-Number": str(chunk_number),
                        "X-Original-Filename": filename
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro no processamento do áudio: {str(e)}")

