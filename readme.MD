Utilizar o process_videos_phantom_v2_ia.py: mais atualizado

Na interface:
- Nucleos: valores inteiros. Deixar em None
- Frames background: quantidade de frames que são utilizados para definir o treshold do background (valor que vai ser subtraído do valor de luminosidade de cada frame)
- Cortes verticas x Cortes horizontais: deve ser múltiplo da resolução do vídeo. Valor padrão: 336 x 252
- Percentil: Se = 75%, irá pegar os 25% que mais se destacarem, necessariamente. Depois vai fazer um filtro.
- Max Gap Frames: Para valor igual a 4: associa ao mesmo evento do frame x ao frame x+4. x+5 será outro evento caso os frames anteriores não sejam associados a raio.
- Folga Frames: Margem de segurança. Se 1, pega 1 frame antes e 1 frame depois para o mesmo evento.

O input deve ser a pasta de imagens, e não o vídeo.

P/U, atualiza o range de frames:
- P: primeiro frame
- U: último frame

C/I/L/B, classificar o raio: 
- C: Cloud-to-Ground
- I: Intra-Cloud
- L: LCC
- B: Brilho (fora da tela)
