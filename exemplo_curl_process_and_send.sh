#!/bin/bash

# Exemplo de como fazer a requisição para process-and-send usando curl
# Substitua "caminho/do/seu/audio.wav" pelo caminho real do seu arquivo de áudio

curl -X POST \
  "http://10.67.57.157:8000/audio/process-and-send?intensity=0.8" \
  -H "Content-Type: multipart/form-data" \
  -F "audio_file=@caminho/do/seu/audio.wav;type=audio/wav" 