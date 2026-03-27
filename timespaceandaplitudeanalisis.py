import cv2
import numpy as np
import cupy as cp
import glob
import os
import time
import plotly.graph_objects as go
import sys

# --- 1. CONFIGURAÇÕES PRINCIPAIS ---
CAMINHO_PASTA_ENTRADA = r"D:\Videos\videos_novos_convertidos\v9.1_FNN_Y20250813H181147.590711000_UTC"
NUM_FRAMES_BACKGROUND = 75
NUMERO_DE_CORTES_VERTICAL = 336

# --- 2. FUNÇÕES DE PROCESSAMENTO (GPU - Com correção de NaN e BUGFIX) ---

def calcular_vetor_referencia_vertical_gpu(arquivos_img, num_frames_bg, num_cortes_v):
    print(f"FASE 1 (GPU): Iniciando calibração com os primeiros {num_frames_bg} frames...")
    if len(arquivos_img) < num_frames_bg:
        raise ValueError(f"Frames insuficientes para calibração.")
    
    # --- CORREÇÃO DE BUG CRÍTICO ---
    # O vetor de soma deve ter o tamanho do número de cortes, não de frames.
    soma_medias_v_gpu = cp.zeros(num_cortes_v, dtype=cp.float32) # <-- BUG CORRIGIDO
    
    for i, caminho_frame in enumerate(arquivos_img[:num_frames_bg]):
        print(f"  Lendo frame de background {i+1}/{num_frames_bg}...", end='\r')
        img_cinza_cpu = cv2.imread(caminho_frame, cv2.IMREAD_GRAYSCALE)
        if img_cinza_cpu is None: continue
        
        img_cinza_gpu = cp.asarray(img_cinza_cpu, dtype=cp.float32)
        cortes_v_gpu = cp.array_split(img_cinza_gpu, num_cortes_v, axis=1)
        
        # Usa nanmean para ser robusto contra pixels corrompidos
        medias_gpu = cp.array([cp.nanmean(corte) for corte in cortes_v_gpu])
        
        soma_medias_v_gpu += medias_gpu
        
    print("\nCalibração na GPU concluída.")
    return soma_medias_v_gpu / num_frames_bg

def analisar_frame_vertical_gpu(img_path, vetor_ref_v_gpu, num_cortes_v):
    img_cinza_cpu = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cinza_cpu is None: return None
    
    img_cinza_gpu = cp.asarray(img_cinza_cpu, dtype=cp.float32)
    cortes_v_gpu = cp.array_split(img_cinza_gpu, num_cortes_v, axis=1)
    
    # Usa nanmean para ser robusto contra pixels corrompidos
    medias_abs_v_gpu = cp.array([cp.nanmean(corte) for corte in cortes_v_gpu])
    lums_rel_v_gpu = medias_abs_v_gpu - vetor_ref_v_gpu
    
    # Se o resultado inteiro for NaN (ex: frame todo corrompido), pule este frame
    if cp.isnan(lums_rel_v_gpu).any():
        print(f"\nAviso: Frame {os.path.basename(img_path)} produziu NaN e será ignorado.")
        return None
    
    return lums_rel_v_gpu

# --- FUNÇÃO DE PLOT 3D ATUALIZADA (Usando PLOTLY) ---
def plotar_grafico_3d_plotly(all_lums_rel_v_cpu, num_cortes_v, total_frames):
    print("\nFASE 3: Preparando dados para visualização 3D com Plotly...")
    
    if not all_lums_rel_v_cpu:
        print("ERRO FATAL: Nenhum dado válido foi processado para o plot.")
        return
        
    Z = np.array(all_lums_rel_v_cpu)
    
    if np.isnan(Z).any():
         print("Aviso: Removendo valores NaN dos dados para o plot.")
         Z = np.nan_to_num(Z)

    # Cria as grades X (Espaço) e Y (Tempo)
    X = np.arange(Z.shape[1]) # Eixo X: 0 até num_cortes_v
    Y = np.arange(Z.shape[0]) # Eixo Y: 0 até total_frames
    
    print("Gerando gráfico de superfície 3D...")
    
    # 1. Cria o objeto de superfície 3D
    fig = go.Figure(data=[
        go.Surface(
            z=Z,  # Dados de amplitude
            x=X,  # Dados de espaço
            y=Y,  # Dados de tempo
            colorscale='Viridis', # O 'viridis' que você queria
            colorbar=dict(title='Luminosidade Relativa'), # A barra de cor (escala)
            showscale=True
        )
    ])

    # 2. Configura a aparência, os eixos e as legendas
    fig.update_layout(
        title='Evolução da Luminosidade Relativa (Espaço x Tempo)',
        scene=dict(
            xaxis_title='Espaço (Índice do Corte)',
            yaxis_title='Tempo (Número do Frame)',
            zaxis_title='Luminosidade (Amplitude)',
            # Controla a proporção dos eixos para uma boa visualização
            aspectratio=dict(x=1.5, y=1.5, z=0.5) 
        ),
        margin=dict(l=65, r=50, b=65, t=90) # Ajusta as margens
    )

    # 3. Salva o gráfico em um arquivo HTML
    output_filename = "grafico_3d_final.html"
    fig.write_html(output_filename)
    
    print("\n--- SUCESSO! ---")
    print(f"Gráfico 3D salvo em: {os.path.abspath(output_filename)}")
    print("Abra este arquivo HTML no seu navegador (Chrome, Edge, etc.) para ver o gráfico interativo.")


# --- BLOCO DE EXECUÇÃO PRINCIPAL (Sem alteração, exceto a chamada da função) ---
if __name__ == "__main__":
    try:
        arquivos_img = sorted(glob.glob(os.path.join(CAMINHO_PASTA_ENTRADA, '*.jpg')))
        total_frames = len(arquivos_img)
        if total_frames == 0:
            raise FileNotFoundError(f"Nenhum arquivo .jpg encontrado em '{CAMINHO_PASTA_ENTRADA}'")
        print(f"{total_frames} frames encontrados.")

        start_time = time.time()
        
        vetor_ref_v_gpu = calcular_vetor_referencia_vertical_gpu(arquivos_img, NUM_FRAMES_BACKGROUND, NUMERO_DE_CORTES_VERTICAL)
        
        print(f"\nFASE 2 (GPU): Coletando métricas de todos os {total_frames} frames...")
        all_lums_rel_vectors_gpu = []
        frames_processados_validos = 0
        
        for i, img_path in enumerate(arquivos_img):
            print(f"  Processando frame {i+1}/{total_frames}...", end='\r')
            lums_rel_v_gpu = analisar_frame_vertical_gpu(img_path, vetor_ref_v_gpu, NUMERO_DE_CORTES_VERTICAL)
            
            if lums_rel_v_gpu is not None:
                all_lums_rel_vectors_gpu.append(lums_rel_v_gpu)
                frames_processados_validos += 1
                
        print(f"\nColeta de métricas na GPU concluída. {frames_processados_validos} frames válidos processados.")
        
        print("Transferindo dados da GPU para a CPU para o plot...")
        all_lums_rel_vectors_cpu = [cp.asnumpy(vec) for vec in all_lums_rel_vectors_gpu]

        processing_time = time.time() - start_time
        print(f"Tempo de processamento na GPU: {processing_time:.2f} segundos.")
        
        # --- MUDANÇA: Chamando a nova função do Plotly ---
        plotar_grafico_3d_plotly(all_lums_rel_vectors_cpu, NUMERO_DE_CORTES_VERTICAL, frames_processados_validos)
        
        print("\nProcesso finalizado.")

    except Exception as e:
        print(f"\nERRO: {e}")

