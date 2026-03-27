import cv2
import numpy as np
import os
import glob
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIGURAÇÕES ---
PASTA_ORIGEM = r"D:\filtro1\dados" 
PASTA_DESTINO = r"D:\filtro1\saida"

# --- PARAMETROS DE "JANELAS MENORES" (Alta Resolução) ---
# Aumentei os valores para que cada "célula" da grade seja menor e mais precisa
NUM_CORTES_V = 672  # Antes era 336 (Dobrado)
NUM_CORTES_H = 504  # Antes era 252 (Dobrado)

# Tamanho mínimo do recorte para a IA não receber lixo (ex: 224x224 pixels)
TAMANHO_MINIMO_CROP = 224

# --- FUNÇÃO WORKER (Núcleo do Processamento) ---
def worker_processar_imagem(args):
    caminho_img, dir_saida = args
    
    try:
        # Lê direto em Grayscale para economizar memória e CPU
        img_gray = cv2.imread(caminho_img, cv2.IMREAD_GRAYSCALE)
        if img_gray is None: return 0
        
        h_img, w_img = img_gray.shape
        
        # 1. Análise Vetorial (Agora com janelas menores/mais cortes)
        # axis=1 corta na vertical (colunas), axis=0 corta na horizontal (linhas)
        cortes_v = np.array_split(img_gray, NUM_CORTES_V, axis=1)
        vetor_v = np.array([np.mean(c) for c in cortes_v])
        
        cortes_h = np.array_split(img_gray, NUM_CORTES_H, axis=0)
        vetor_h = np.array([np.mean(c) for c in cortes_h])
        
        # 2. Mapa Térmico
        # Multiplica os vetores. Se houver brilho em X e Y, o ponto "acende".
        heatmap = np.outer(vetor_h, vetor_v)
        
        # Segurança: Se a imagem for totalmente preta, heatmap.max() é 0.
        max_heat = heatmap.max()
        if max_heat == 0:
            return 0 
            
        # Normaliza para 0-255
        heatmap_norm = (heatmap / max_heat * 255).astype(np.uint8)
        
        # 3. Definição da Área (Limiar Baixo = Mais sensível)
        # Baixei para 20. Se tiver 20/255 de brilho relativo, ele pega.
        _, mask = cv2.threshold(heatmap_norm, 20, 255, cv2.THRESH_BINARY)
        
        pontos_y, pontos_x = np.where(mask > 0)
        
        if len(pontos_x) == 0:
            return 0
            
        # Converte coordenadas da matriz térmica para pixels reais da imagem
        escala_x = w_img / NUM_CORTES_V
        escala_y = h_img / NUM_CORTES_H
        
        min_x = int(np.min(pontos_x) * escala_x)
        max_x = int(np.max(pontos_x) * escala_x)
        min_y = int(np.min(pontos_y) * escala_y)
        max_y = int(np.max(pontos_y) * escala_y)
        
        # Centro do objeto detectado
        center_x = (min_x + max_x) // 2
        center_y = (min_y + max_y) // 2
        
        # Largura e Altura atuais da detecção
        curr_w = max_x - min_x
        curr_h = max_y - min_y
        
        # 4. Forçar Tamanho Mínimo (Para a IA ver contexto)
        # Se o recorte for menor que TAMANHO_MINIMO_CROP, expandimos ao redor do centro
        target_w = max(curr_w + 50, TAMANHO_MINIMO_CROP) # +50 de margem
        target_h = max(curr_h + 50, TAMANHO_MINIMO_CROP)
        
        # Recalcula coordenadas finais baseadas no centro
        final_x1 = max(0, center_x - target_w // 2)
        final_y1 = max(0, center_y - target_h // 2)
        final_x2 = min(w_img, center_x + target_w // 2)
        final_y2 = min(h_img, center_y + target_h // 2)
        
        # Recorta
        # Nota: Carregamos em Gray para processar, mas se quiser salvar colorido
        # teria que carregar BGR. Para treinar IA de raios, Gray costuma ser suficiente e mais rápido.
        # Se precisar colorido, mude o imread lá em cima para BGR e faça cvtColor depois.
        crop = img_gray[final_y1:final_y2, final_x1:final_x2]
        
        if crop.size > 0:
            nome_original = os.path.basename(caminho_img)
            nome_novo = f"crop_{nome_original}"
            caminho_final = os.path.join(dir_saida, nome_novo)
            cv2.imwrite(caminho_final, crop)
            return 1
            
        return 0
        
    except Exception as e:
        # Se der erro em um arquivo, ignora e segue (não trava o processo)
        return 0

# --- LOOP PRINCIPAL ---
def main():
    # Setup Pastas
    path_raios = os.path.join(PASTA_DESTINO, "RAIO")
    # path_nao_raios = os.path.join(PASTA_DESTINO, "NAO_RAIO") 
    os.makedirs(path_raios, exist_ok=True)
    
    print(f"--- PROCESSADOR TURBO DE DATASET ---")
    num_cores = multiprocessing.cpu_count()
    print(f"Detectado: {num_cores} núcleos lógicos.")
    print(f"Resolução da Análise: {NUM_CORTES_V}x{NUM_CORTES_H} (Janelas Menores Ativadas)")
    
    print(f"Buscando imagens em: {PASTA_ORIGEM}...")
    arquivos = glob.glob(os.path.join(PASTA_ORIGEM, "**", "*.jpg"), recursive=True)
    
    total = len(arquivos)
    print(f"Total de imagens na fila: {total}")
    
    if total == 0:
        return

    # Lista de tarefas
    # Ajuste o destino aqui se estiver processando a pasta de 'NAO_RAIOS'
    args_list = [(f, path_raios) for f in arquivos]
    
    sucessos = 0
    start = time.time()
    
    print("Iniciando motores...")
    
    with ProcessPoolExecutor(max_workers=num_cores) as executor:
        # map distribui as imagens entre os núcleos do processador
        resultados = executor.map(worker_processar_imagem, args_list)
        
        for i, res in enumerate(resultados):
            sucessos += res
            if i % 50 == 0: # Atualiza a cada 50 fotos
                perc = (i / total) * 100
                print(f" Progresso: {perc:.1f}% | Salvos: {sucessos} ", end='\r')

    tempo = time.time() - start
    fps = total / tempo if tempo > 0 else 0
    
    print(f"\n\n--- FINALIZADO ---")
    print(f"Tempo: {tempo:.2f}s | Velocidade: {fps:.1f} imgs/s")
    print(f"Recortes gerados: {sucessos} de {total}")
    print(f"Salvo em: {path_raios}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()