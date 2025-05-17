import io
import torch
import torchaudio
import os

N_FFT = 512
HOP_LENGTH = 128
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importar seu modelo aqui (ajuste conforme seu projeto)
from app.model.model import UNetDenoiser

BASE_DIR = os.path.dirname(os.path.dirname(__file__))  # isso vai para 'app/'
MODEL_PATH = os.path.join(BASE_DIR, "model", "best_denoiser_model.pth")

# Inicializa e carrega o modelo uma vez
model = UNetDenoiser().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

import torch.nn.functional as F

N_FFT = 512
HOP_LENGTH = 128
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importar seu modelo UNet
from app.model.model import UNetDenoiser

# Inicializa e carrega o modelo (ajuste o caminho se necessário)
model = UNetDenoiser().to(DEVICE)
model.load_state_dict(torch.load("app/model/best_denoiser_model.pth", map_location=DEVICE))
model.eval()


def reconstruct_audio(noisy_spec, mask, hop_length=HOP_LENGTH, original_length=None):
    # noisy_spec['magnitude']: Tensor 2D [freq_bins, time_frames]
    magnitude = torch.exp(noisy_spec['magnitude']) - 1

    # mask: Tensor 2D [freq_bins, time_frames]
    enhanced_magnitude = magnitude * mask

    # Reconstruir espectrograma complexo
    enhanced_stft = enhanced_magnitude * torch.exp(1j * noisy_spec['phase'])

    window = torch.hann_window(N_FFT, device=enhanced_stft.device)
    length = original_length

    audio = torch.istft(
        enhanced_stft,
        n_fft=N_FFT,
        hop_length=hop_length,
        window=window,
        length=length
    )
    return audio


def denoise_waveform(waveform, sr):
    # ### 1) Converter para mono
    if waveform.dim() == 2 and waveform.size(0) > 1:
        waveform = torch.mean(waveform, dim=0, keepdim=True)  # [1, samples]

    # ### 2) Reamostrar se necessário
    if sr != SAMPLE_RATE:
        waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)
    waveform = waveform.to(DEVICE)  # [1, samples]

    # ### 3) STFT
    # Passe 1D para stft: waveform[0]
    window = torch.hann_window(N_FFT).to(DEVICE)
    stft = torch.stft(
        waveform[0],
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        window=window,
        return_complex=True
    )  # shape [freq_bins, time_frames]

    magnitude = torch.log1p(torch.abs(stft))  # [freq_bins, time_frames]
    phase     = torch.angle(stft)              # [freq_bins, time_frames]

    # ### 4) Preparar input para o modelo (4D)
    noisy_mag = magnitude.unsqueeze(0).unsqueeze(0)  # [1, 1, freq, time]
    print("[DEBUG] noisy_mag shape:", noisy_mag.shape)

    # ### 5) Inferência
    with torch.no_grad():
        mask = model(noisy_mag)  # [1, 1, freq, time]
    print("[DEBUG] mask shape:", mask.shape,
          "min/max/mean:", mask.min().item(), mask.max().item(), mask.mean().item())

    # ### 6) Squeeze para 2D
    mask_2d = mask.squeeze(0).squeeze(0)  # [freq, time]

    # ### 7) Reconstruir áudio usando ISTFT
    denoised = reconstruct_audio(
        {'magnitude': magnitude, 'phase': phase},
        mask_2d,
        hop_length=HOP_LENGTH,
        original_length=waveform.size(1)
    )

    return denoised.cpu()




# Exemplo de uso: (suponha que você carregou um waveform e sample rate)
# waveform, sr = torchaudio.load("seuarquivo.wav")
# denoised = denoise_waveform(waveform[0], sr)  # waveform[0] para mono



# def reconstruct_audio(noisy_spec, mask, hop_length=HOP_LENGTH):
#     magnitude = torch.exp(noisy_spec['magnitude']) - 1  # Reverter log1p
#     enhanced_magnitude = magnitude * mask.squeeze(1)  # [1, freq, time]

#     # Se necessário, remova dimensões extras
#     enhanced_magnitude = enhanced_magnitude.squeeze(0)  # [freq, time]
#     phase = noisy_spec['phase'].squeeze(0)  # [freq, time]

#     enhanced_stft = enhanced_magnitude * torch.exp(1j * phase)  # [freq, time]

#     audio = torch.istft(
#         enhanced_stft,
#         n_fft=N_FFT,
#         hop_length=hop_length,
#         window=torch.hann_window(N_FFT, device=enhanced_stft.device),
#         length=None
#     )
#     return audio

# def denoise_waveform(waveform, sr):
#     if sr != SAMPLE_RATE:
#         waveform = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(waveform)

#     if waveform.shape[0] > 1:
#         waveform = torch.mean(waveform, dim=0, keepdim=True)  # [1, samples]

#     waveform = waveform.to(DEVICE)

#     window = torch.hann_window(N_FFT).to(DEVICE)
#     stft = torch.stft(
#         waveform, n_fft=N_FFT, hop_length=HOP_LENGTH,
#         window=window, return_complex=True
#     )

#     magnitude = torch.log1p(torch.abs(stft))  # [1, freq, time]
#     phase = torch.angle(stft)

#     noisy_mag = magnitude.unsqueeze(0)  # [1, 1, freq, time]

#     print(">>> [DEBUG] waveform:", waveform.shape)
#     print(">>> [DEBUG] magnitude:", magnitude.shape)
#     print(">>> [DEBUG] noisy_mag (input to model):", noisy_mag.shape)

#     with torch.no_grad():
#         mask = model(noisy_mag)

#     print(">>> [DEBUG] mask (output from model):", mask.shape)

#     denoised_audio = reconstruct_audio({'magnitude': magnitude, 'phase': phase}, mask, hop_length=HOP_LENGTH)
#     return denoised_audio.cpu()


