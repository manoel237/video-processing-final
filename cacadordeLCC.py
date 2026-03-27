import cv2
import numpy as np
import glob
import os
import shutil
from datetime import datetime

# --- FUNÇÃO AUXILIAR PARA ANÁLISE DE ESTRUTURA ---
def encontrar_maior_sequencia_contigua(array_booleano):
    """Encontra o comprimento da maior sequência de valores True em um array."""
    if not array_booleano.any():
        return 0
    maior_contagem = 0
    contagem_atual = 0
    for valor in array_booleano:
        if valor:
            contagem_atual += 1
        else:
            maior_contagem = max(maior_contagem, contagem_atual)
            contagem_atual = 0
    return max(maior_contagem, contagem_atual)

# --- PARÂMETROS DE CONFIGURAÇÃO ---
# MODIFICADO: Agora é a pasta que contém TODAS as pastas de vídeos
PASTA_RAIZ_ENTRADAS = r"C:\Arquivos\Videos\intest"
PASTA_RAIZ_SAIDAS = r"C:\Arquivos\Videos\outcronos"
NOME_ARQUIVO_RELATORIO = "relatorio_LCCstestando.txt"
TEMPO_DE_GRAVACAO_SEGUNDOS = 1.266 # Usado para calcular a duração em ms

# --- Parâmetros de Análise ---
NUM_FRAMES_BACKGROUND = 250
NUMERO_DE_CORTES_VERTICAL = 320
NUMERO_DE_CORTES_HORIZONTAL = 256

# --- Parâmetros de Detecção ---
PERCENTIL = 60
USAR_ANALISE_CONTIGUIDADE = True
MIN_FATIAS_CONTIGUAS = 15
LIMIAR_BRILHO_CONTIGUIDADE = 0.25

# --- PARÂMETROS DE FILTRAGEM DE EVENTOS LONGOS (LCC) ---
USAR_FILTRO_LCC = True
DURACAO_MINIMA_LCC = 100

# --- Parâmetros de Agrupamento ---
MAX_GAP_ENTRE_FRAMES = 20
FOLGA_FRAMES = 5

def calcular_vetores_referencia(arquivos_img, num_frames_bg, num_cortes_v, num_cortes_h):
    """Calcula a luminosidade de fundo média para os cortes verticais e horizontais."""
    if len(arquivos_img) < num_frames_bg:
        print(f"   AVISO: Frames insuficientes para calibração ({len(arquivos_img)} de {num_frames_bg}). Pulando vídeo.")
        return None, None
    
    frames_para_bg = arquivos_img[:num_frames_bg]
    soma_medias_v = np.zeros(num_cortes_v)
    soma_medias_h = np.zeros(num_cortes_h)
    
    for i, caminho_frame in enumerate(frames_para_bg):
        img_cinza = cv2.imread(caminho_frame, cv2.IMREAD_GRAYSCALE)
        if img_cinza is None: continue
        
        cortes_v = np.array_split(img_cinza, num_cortes_v, axis=1)
        soma_medias_v += np.array([np.mean(corte) for corte in cortes_v])
        
        cortes_h = np.array_split(img_cinza, num_cortes_h, axis=0)
        soma_medias_h += np.array([np.mean(corte) for corte in cortes_h])
        
    return (soma_medias_v / num_frames_bg), (soma_medias_h / num_frames_bg)

def analisar_frame(img_path, vetor_ref_v, vetor_ref_h, num_cortes_v, num_cortes_h):
    """Calcula as métricas globais e a nova métrica de contiguidade para um frame."""
    img_cinza = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cinza is None: return None
    
    cortes_v = np.array_split(img_cinza, num_cortes_v, axis=1)
    medias_abs_v = [np.mean(corte) for corte in cortes_v]
    lums_rel_v = np.array(medias_abs_v) - vetor_ref_v
    dp_v = np.std(lums_rel_v)
    vmax_v = np.max(lums_rel_v)

    contiguidade_v = 0
    if USAR_ANALISE_CONTIGUIDADE:
        fatias_ativas_v = lums_rel_v > LIMIAR_BRILHO_CONTIGUIDADE
        contiguidade_v = encontrar_maior_sequencia_contigua(fatias_ativas_v)

    cortes_h = np.array_split(img_cinza, num_cortes_h, axis=0)
    medias_abs_h = [np.mean(corte) for corte in cortes_h]
    lums_rel_h = np.array(medias_abs_h) - vetor_ref_h
    dp_h = np.std(lums_rel_h)
    vmax_h = np.max(lums_rel_h)

    contiguidade_h = 0
    if USAR_ANALISE_CONTIGUIDADE:
        fatias_ativas_h = lums_rel_h > LIMIAR_BRILHO_CONTIGUIDADE
        contiguidade_h = encontrar_maior_sequencia_contigua(fatias_ativas_h)
    
    return (dp_v, vmax_v, contiguidade_v), (dp_h, vmax_h, contiguidade_h)

def agrupar_em_eventos(indices_chave, max_gap, folga, total_frames):
    if not indices_chave: return []
    eventos = []
    evento_atual = [indices_chave[0]]
    for i in range(1, len(indices_chave)):
        if indices_chave[i] - evento_atual[-1] <= max_gap + 1:
            evento_atual.append(indices_chave[i])
        else:
            inicio_evento = max(0, min(evento_atual) - folga)
            fim_evento = min(total_frames - 1, max(evento_atual) + folga)
            eventos.append((inicio_evento, fim_evento))
            evento_atual = [indices_chave[i]]
    inicio_evento = max(0, min(evento_atual) - folga)
    fim_evento = min(total_frames - 1, max(evento_atual) + folga)
    eventos.append((inicio_evento, fim_evento))
    return eventos

# --- NOVA FUNÇÃO PRINCIPAL DE PROCESSAMENTO ---
def processar_pasta_video(caminho_video):
    """Executa todo o pipeline de detecção para uma única pasta de vídeo e retorna os LCCs encontrados."""
    arquivos_img = sorted(glob.glob(os.path.join(caminho_video, '*.jpg')))
    total_frames = len(arquivos_img)
    if total_frames == 0:
        print("   Nenhum frame .jpg encontrado. Pulando.")
        return []

    print(f"   Analisando {total_frames} frames...")
    
    tempo_por_frame_ms = (TEMPO_DE_GRAVACAO_SEGUNDOS * 1000) / total_frames
    
    vetor_ref_v, vetor_ref_h = calcular_vetores_referencia(
        arquivos_img, NUM_FRAMES_BACKGROUND, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL
    )
    if vetor_ref_v is None:
        return []

    all_metrics_v = []
    all_metrics_h = []
    for img_path in arquivos_img:
        result = analisar_frame(
            img_path, vetor_ref_v, vetor_ref_h, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL
        )
        if result:
            all_metrics_v.append(result[0])
            all_metrics_h.append(result[1])
    
    if not all_metrics_v:
        return []

    all_std_devs_v, all_max_vals_v, all_contig_v = zip(*all_metrics_v)
    all_std_devs_h, all_max_vals_h, all_contig_h = zip(*all_metrics_h)
    
    limiar_dp_v = np.percentile(all_std_devs_v, PERCENTIL)
    limiar_pico_v = np.percentile(all_max_vals_v, PERCENTIL)
    limiar_dp_h = np.percentile(all_std_devs_h, PERCENTIL)
    limiar_pico_h = np.percentile(all_max_vals_h, PERCENTIL)
    
    frames_chave_indices = []
    for i in range(len(all_std_devs_v)):
        passou_sensibilidade_global = (all_std_devs_v[i] > limiar_dp_v or all_max_vals_v[i] > limiar_pico_v) or \
                                     (all_std_devs_h[i] > limiar_dp_h or all_max_vals_h[i] > limiar_pico_h)
        if not passou_sensibilidade_global:
            continue
        if USAR_ANALISE_CONTIGUIDADE:
            tem_estrutura_minima = all_contig_v[i] >= MIN_FATIAS_CONTIGUAS or \
                                     all_contig_h[i] >= MIN_FATIAS_CONTIGUAS
            if tem_estrutura_minima:
                frames_chave_indices.append(i)
        else:
            frames_chave_indices.append(i)
    
    eventos_candidatos = agrupar_em_eventos(frames_chave_indices, MAX_GAP_ENTRE_FRAMES, FOLGA_FRAMES, total_frames)
    
    if USAR_FILTRO_LCC:
        eventos_filtrados = []
        for evento in eventos_candidatos:
            inicio, fim = evento
            duracao = fim - inicio + 1
            if duracao >= DURACAO_MINIMA_LCC:
                eventos_filtrados.append(evento)
        eventos_candidatos = eventos_filtrados
    
    # Prepara a lista de resultados para este vídeo
    resultados_finais = []
    for evento in eventos_candidatos:
        inicio, fim = evento
        duracao_frames = fim - inicio + 1
        duracao_ms = duracao_frames * tempo_por_frame_ms
        resultados_finais.append(
            (f"[{inicio} - {fim}]", f"{duracao_ms:.1f} ms")
        )
        
    return resultados_finais

# --- Bloco de Execução Principal (MODO BATCH) ---
if __name__ == "__main__":
    print("--- INICIANDO ANÁLISE EM MASSA DE VÍDEOS PARA LCC ---")
    
    # Encontra todas as subpastas na pasta de entrada
    try:
        lista_de_videos = [f.path for f in os.scandir(PASTA_RAIZ_ENTRADAS) if f.is_dir()]
    except FileNotFoundError:
        print(f"ERRO: A pasta de entrada '{PASTA_RAIZ_ENTRADAS}' não foi encontrada.")
        exit()

    if not lista_de_videos:
        print(f"Nenhuma subpasta de vídeo encontrada em '{PASTA_RAIZ_ENTRADAS}'.")
        exit()
        
    print(f"{len(lista_de_videos)} pastas de vídeo encontradas. Iniciando processamento...")
    
    videos_com_lcc = {}

    for i, caminho_video in enumerate(lista_de_videos):
        nome_video = os.path.basename(caminho_video)
        print(f"\n--- Processando Vídeo {i+1}/{len(lista_de_videos)}: {nome_video} ---")
        
        try:
            resultados_lcc = processar_pasta_video(caminho_video)
            if resultados_lcc:
                videos_com_lcc[nome_video] = resultados_lcc
        except Exception as e:
            print(f"   ERRO INESPERADO ao processar o vídeo {nome_video}: {e}")

    # --- Geração do Relatório Final ---
    caminho_relatorio = os.path.join(PASTA_RAIZ_SAIDAS, NOME_ARQUIVO_RELATORIO)
    os.makedirs(PASTA_RAIZ_SAIDAS, exist_ok=True)
    
    print("\n--- GERAÇÃO DO RELATÓRIO ---")
    with open(caminho_relatorio, 'w', encoding='utf-8') as f:
        f.write(f"Relatório de Detecção de LCCs (Long Continuous Current)\n")
        f.write(f"Gerado em: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Pasta Analisada: {PASTA_RAIZ_ENTRADAS}\n")
        f.write("="*50 + "\n\n")

        if not videos_com_lcc:
            f.write("Nenhum vídeo com possíveis LCCs foi encontrado com os parâmetros atuais.\n")
            print("Nenhum LCC encontrado.")
        else:
            print(f"{len(videos_com_lcc)} vídeo(s) com possíveis LCCs foram encontrados.")
            for nome_video, eventos in videos_com_lcc.items():
                f.write(f"Vídeo: {nome_video}\n")
                f.write(f"  - Quantidade de Possíveis LCCs: {len(eventos)}\n")
                for j, (intervalo, duracao) in enumerate(eventos):
                    f.write(f"    - LCC #{j+1}:\n")
                    f.write(f"      - Intervalo de Frames: {intervalo}\n")
                    f.write(f"      - Duração Estimada:    {duracao}\n")
                f.write("\n")
    
    print(f"\nRelatório final salvo em: {caminho_relatorio}")
    print("Análise em massa concluída.")