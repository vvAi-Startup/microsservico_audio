import os
import torch
import torchaudio
import numpy as np
from typing import Tuple, Optional
import datetime
import shutil

N_FFT = 512
HOP_LENGTH = 128
SAMPLE_RATE = 16000
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Importar a definição do modelo do arquivo de treinamento
from app.model.model import UNetDenoiser
from app.utils.reconstruct_audio import reconstruct_audio

class AudioService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = UNetDenoiser().to(self.device)
        self.output_dir = os.path.join(os.path.dirname(__file__), "..", "audio_files")
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Carregar modelo
        try:
            model_path = os.path.join(os.path.dirname(__file__), "..", "model", "best_denoiser_model.pth")
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            self.model.eval()
            print("Modelo carregado com sucesso!")
        except Exception as e:
            print(f"Erro ao carregar o modelo: {e}")
            raise


    async def process_audio(self, input_file: str, intensity: float = 1.0) -> Tuple[str, str]:
        """
        Processa um arquivo de áudio para remover ruído.
        
        Args:
            input_file: Caminho do arquivo de entrada
            intensity: Intensidade do processamento (0.0 a 1.0)
            
        Returns:
            Tuple[str, str]: (caminho do arquivo processado, diretório do processo)
        """
        try:
            # Criar diretório para o processamento atual
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            process_dir = os.path.join(self.output_dir, f"process_{timestamp}")
            os.makedirs(process_dir, exist_ok=True)
            
            # Remover a cópia do arquivo original
            # original_filename = os.path.basename(input_file)
            # original_copy_path = os.path.join(process_dir, "original_" + original_filename)
            # shutil.copy2(input_file, original_copy_path)
            
            # Carregar áudio
            audio, sr = torchaudio.load(input_file)
            
            # Converter para mono se estiver em estéreo
            if audio.shape[0] > 1:
                audio = torch.mean(audio, dim=0, keepdim=True)
            
            # Reamostrar se necessário
            if sr != SAMPLE_RATE:
                audio = torchaudio.transforms.Resample(sr, SAMPLE_RATE)(audio)
            
            # Normalizar
            audio = audio / (torch.max(torch.abs(audio)) + 1e-8)
            
            # Calcular STFT
            stft = torch.stft(
                audio.squeeze(0), 
                n_fft=N_FFT, 
                hop_length=HOP_LENGTH, 
                window=torch.hann_window(N_FFT), 
                return_complex=True
            )
            
            # Converter para magnitude e fase
            mag = torch.abs(stft)
            phase = torch.angle(stft)
            
            # Aplicar escala logarítmica à magnitude
            log_mag = torch.log1p(mag)
            
            # Processar em lotes para evitar problemas de memória
            batch_size = 10  # segundos
            samples_per_batch = batch_size * SAMPLE_RATE
            hop_samples = int(samples_per_batch * 0.5)  # 50% de sobreposição
            
            audio_length = audio.shape[1]
            num_batches = max(1, int(np.ceil(audio_length / hop_samples)))
            
            # Preparar áudio processado
            processed_audio = torch.zeros_like(audio)
            overlap_count = torch.zeros_like(audio)
            
            for i in range(num_batches):
                start_sample = i * hop_samples
                end_sample = min(start_sample + samples_per_batch, audio_length)
                
                # Extrair segmento
                segment = audio[:, start_sample:end_sample]
                
                # Calcular STFT do segmento
                segment_stft = torch.stft(
                    segment.squeeze(0), 
                    n_fft=N_FFT, 
                    hop_length=HOP_LENGTH, 
                    window=torch.hann_window(N_FFT), 
                    return_complex=True
                )
                
                segment_mag = torch.abs(segment_stft)
                segment_phase = torch.angle(segment_stft)
                segment_log_mag = torch.log1p(segment_mag)
                
                # Preparar para o modelo
                noisy_spec = segment_log_mag.unsqueeze(0).unsqueeze(0).to(self.device)
                
                # Aplicar modelo
                with torch.no_grad():
                    mask = self.model(noisy_spec)
                
                # Aplicar intensidade ao mask
                if intensity < 1.0:
                    unity_mask = torch.ones_like(mask)
                    mask = intensity * mask + (1 - intensity) * unity_mask
                
                # Reconstruir áudio
                denoised_segment = reconstruct_audio(
                    {'magnitude': noisy_spec.squeeze(1), 'phase': segment_phase.unsqueeze(0).to(self.device)},
                    mask,
                    HOP_LENGTH
                )
                
                # Garantir que denoised_segment tenha a forma correta
                if len(denoised_segment.shape) == 1:
                    denoised_segment = denoised_segment.unsqueeze(0)
                
                # Verificar se o comprimento do segmento é compatível
                if denoised_segment.shape[0] != 1:
                    denoised_segment = denoised_segment.unsqueeze(0)
                
                # Adicionar ao áudio processado com sobreposição
                segment_length = denoised_segment.shape[1]
                if start_sample + segment_length <= processed_audio.shape[1]:
                    processed_audio[0, start_sample:start_sample + segment_length] += denoised_segment[0]
                    overlap_count[0, start_sample:start_sample + segment_length] += 1
                else:
                    available_length = processed_audio.shape[1] - start_sample
                    processed_audio[0, start_sample:] += denoised_segment[0, :available_length]
                    overlap_count[0, start_sample:] += 1
            
            # Normalizar pela contagem de sobreposição
            processed_audio = processed_audio / (overlap_count + 1e-8)
            
            # Normalizar amplitude
            processed_audio = processed_audio / (torch.max(torch.abs(processed_audio)) + 1e-8)
            
            # Salvar resultado
            output_file = os.path.join(process_dir, "denoised_" + os.path.basename(input_file))
            torchaudio.save(output_file, processed_audio, SAMPLE_RATE)
            
            # Criar arquivo de informações
            # info_file = os.path.join(process_dir, "info.txt")
            # with open(info_file, "w") as f:
            #     f.write(f"Processamento realizado em: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            #     f.write(f"Arquivo original: {input_file}\n")
            #     f.write(f"Duração: {audio_length / SAMPLE_RATE:.2f} segundos\n")
            #     f.write(f"Taxa de amostragem: {SAMPLE_RATE} Hz\n")
            #     f.write(f"Dispositivo usado: {self.device}\n")
            #     f.write(f"Intensidade do processamento: {intensity:.2f}\n")
            
            return output_file, process_dir
            
        except Exception as e:
            print(f"Erro no processamento: {e}")
            import traceback
            traceback.print_exc()
            raise 