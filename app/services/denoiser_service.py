import torch
import torchaudio
from io import BytesIO

# Carregar o modelo uma vez só
MODEL_PATH = "best_denoiser_model.pth"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Carrega o modelo (ajustado conforme sua classe de modelo)
model = torch.load(MODEL_PATH, map_location=device)
model.eval()

def denoise_audio(file_bytes: bytes) -> bytes:
    # Carrega o áudio do arquivo em memória
    audio_tensor, sample_rate = torchaudio.load(BytesIO(file_bytes))

    # Envia para o dispositivo
    audio_tensor = audio_tensor.to(device)

    # Garante que está no formato que o modelo espera
    with torch.no_grad():
        denoised = model(audio_tensor.unsqueeze(0))  # Ex: adiciona batch dim
        denoised = denoised.squeeze(0).cpu()

    # Salva em memória como WAV
    buffer = BytesIO()
    torchaudio.save(buffer, denoised, sample_rate, format="wav")
    buffer.seek(0)

    return buffer.read()
