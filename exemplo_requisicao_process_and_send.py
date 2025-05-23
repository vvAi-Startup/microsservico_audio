import requests

def processar_e_enviar_audio(caminho_arquivo, intensidade=1.0):
    """
    Processa um arquivo de áudio e envia automaticamente para o servidor especificado.
    
    Args:
        caminho_arquivo (str): Caminho do arquivo de áudio a ser processado
        intensidade (float): Intensidade do processamento (0.0 a 1.0)
    
    Returns:
        dict: Resposta do servidor
    """
    # URL do endpoint
    url = "http://10.67.57.148:8000/audio/process-and-send"
    
    # Preparar os parâmetros da requisição
    params = {
        "intensity": intensidade
    }
    
    # Preparar o arquivo para envio
    with open(caminho_arquivo, "rb") as arquivo:
        files = {
            "audio_file": (arquivo.name, arquivo, "audio/wav")
        }
        
        # Fazer a requisição
        response = requests.post(url, params=params, files=files)
        
        # Verificar se a requisição foi bem-sucedida
        if response.status_code == 200:
            print("Sucesso! Áudio processado e enviado para o servidor.")
            return response.json()
        else:
            print(f"Erro na requisição: {response.status_code}")
            print(f"Detalhes: {response.text}")
            return None

# Exemplo de uso
if __name__ == "__main__":
    # Exemplo básico
    caminho_audio = "exemplo.wav"  # Substitua pelo caminho do seu arquivo de áudio
    resultado = processar_e_enviar_audio(
        caminho_arquivo=caminho_audio,
        intensidade=0.8
    ) 