import torch

from app.services.denoiser_service import HOP_LENGTH, N_FFT

def reconstruct_audio(noisy_spec, mask, hop_length=HOP_LENGTH):
    # Aplicar a máscara ao espectrograma ruidoso
    magnitude = torch.exp(noisy_spec['magnitude']) - 1  # Reverter log1p
    enhanced_magnitude = magnitude * mask.squeeze(1)
    
    # Recombinar magnitude e fase para voltar ao domínio complexo
    enhanced_stft = enhanced_magnitude * torch.exp(1j * noisy_spec['phase'])
    
    # Reconstruir o sinal de áudio usando a ISTFT
    audio = torch.istft(
        enhanced_stft, 
        n_fft=N_FFT, 
        hop_length=hop_length, 
        window=torch.hann_window(N_FFT, device=enhanced_stft.device), 
        length=None  # Removido SEGMENT_LENGTH pois não é necessário para inferência
    )
    
    return audio  
