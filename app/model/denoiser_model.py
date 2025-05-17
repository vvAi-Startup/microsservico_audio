import torch
import torch.nn as nn
import torch.nn.functional as F

# Configurações
SAMPLE_RATE = 16000
N_FFT = 512
HOP_LENGTH = 128

class UNetDenoiser(nn.Module):
    def __init__(self):
        super(UNetDenoiser, self).__init__()
        
        # Encoder (Downsampling)
        self.enc1 = self._encoder_block(1, 16)
        self.enc2 = self._encoder_block(16, 32)
        self.enc3 = self._encoder_block(32, 64)
        self.enc4 = self._encoder_block(64, 128)
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5)
        )
        
        # Decoder (Upsampling)
        self.dec4 = self._decoder_block(256 + 128, 128)
        self.dec3 = self._decoder_block(128 + 64, 64)
        self.dec2 = self._decoder_block(64 + 32, 32)
        self.dec1 = self._decoder_block(32 + 16, 16)
        
        # Camada final para gerar a máscara
        self.final = nn.Sequential(
            nn.Conv2d(16, 1, kernel_size=1),
            nn.Sigmoid()  # Sigmoid para garantir que os valores da máscara estejam entre 0 e 1
        )
        
    def _encoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d(2)
        )
    
    def _decoder_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=2, stride=2)
        )
    
    def forward(self, x):
        # Dimensões esperadas: (batch_size, 1, freq_bins, time_frames)
        
        # Encoder
        enc1_out = self.enc1(x)
        enc2_out = self.enc2(enc1_out)
        enc3_out = self.enc3(enc2_out)
        enc4_out = self.enc4(enc3_out)
        
        # Bottleneck
        bottleneck_out = self.bottleneck(enc4_out)
        
        # Decoder com skip connections
        # Adicionar verificação e redimensionamento para garantir dimensões compatíveis
        dec4_out = self.dec4(torch.cat([bottleneck_out, self._crop_tensor(enc4_out, bottleneck_out)], dim=1))
        dec3_out = self.dec3(torch.cat([dec4_out, self._crop_tensor(enc3_out, dec4_out)], dim=1))
        dec2_out = self.dec2(torch.cat([dec3_out, self._crop_tensor(enc2_out, dec3_out)], dim=1))
        dec1_out = self.dec1(torch.cat([dec2_out, self._crop_tensor(enc1_out, dec2_out)], dim=1))
        
        # Camada final para gerar a máscara
        mask = self.final(dec1_out)
        
        # Garantir que a máscara tenha o mesmo tamanho da entrada
        if mask.size() != x.size():
            mask = F.interpolate(mask, size=(x.size(2), x.size(3)), mode='bilinear', align_corners=False)
        
        return mask
    
    def _crop_tensor(self, source, target):
        # Recorta o tensor fonte para corresponder às dimensões do tensor alvo
        _, _, h_source, w_source = source.size()
        _, _, h_target, w_target = target.size()
        
        h_diff = (h_source - h_target) // 2
        w_diff = (w_source - w_target) // 2
        
        if h_diff < 0 or w_diff < 0:
            # Se o alvo for maior que a fonte, redimensione a fonte
            return F.interpolate(source, size=(h_target, w_target), mode='bilinear', align_corners=False)
        
        # Caso contrário, recorte a fonte
        return source[:, :, h_diff:h_diff+h_target, w_diff:w_diff+w_target]

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