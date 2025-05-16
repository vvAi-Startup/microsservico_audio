from fastapi import APIRouter, UploadFile, File, Form, HTTPException
from fastapi.responses import StreamingResponse
import io
import torchaudio
from app.services.denoiser_service import denoise_waveform, SAMPLE_RATE

torchaudio.set_audio_backend("soundfile")

router = APIRouter()

@router.post("/denoise")
async def process_audio(
    audio_file: UploadFile = File(...),
    session_id: str = Form(...),
    user_id: str = Form(...),
    chunk_number: int = Form(...),
    filename: str = Form(...)
):
    
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

