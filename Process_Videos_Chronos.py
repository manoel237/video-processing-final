import cv2
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

# Algumas considerações: Esse código foi personalizado para funcionar com os dados da Chronos, logo pode não ter um bom desempenho para
# os videos da Phantom ou outra câmera com comportamentos diferentes;

# --- PARÂMETROS DE CONFIGURAÇÃO (Fixo para Cronos) ---
print("--- Câmera CRONOS selecionada ---")
# PASTA DE FRAMES EXTRAÍDOS DO VÍDEO
CAMINHO_PASTA_ENTRADA = r"D:\videos\vid_2025-09-23\2025-09-23-01-25-06-807268700\frames-2025-09-23-01-25-06-807268700"

# PASTA RAIZ ONDE AS PASTAS CLASSIFICADAS SERÃO SALVAS
PASTA_RAIZ_SAIDAS     = r"D:\videos\RESULTADOS"

SUFIXO_PASTA_SAIDA     = "_classificado"
TEMPO_DE_GRAVACAO_SEGUNDOS = 1.266

# --- Parâmetros de Análise ---
# Importante: A quantidade de cortes feita precisa ser um número divisível pela resolução da imagem
# isso vale horizontalmente e verticalmente
NUM_FRAMES_BACKGROUND = 250
NUMERO_DE_CORTES_VERTICAL = 320
NUMERO_DE_CORTES_HORIZONTAL = 256

# Sensibilidade da classificação - Quanto maior menos sensível; Recomendado = 80 ou menos
PERCENTIL = 98 # Quanto maior mais seletivo é o sistema

# Esse parâmetro é crucial para pegar casos difíceis, de baixa amplitude e com picos mal definidos
# No entanto pode acabar gerando um número muito maior de possíveis casos, muitos deles falsos casos
USAR_ANALISE_CONTIGUIDADE = True
MIN_FATIAS_CONTIGUAS = 20          
LIMIAR_BRILHO_CONTIGUIDADE = 0.1

# --- Parâmetros de Agrupamento --- Esses valores foram obtidos a mão e assim conseguimos separar os eventos
MAX_GAP_ENTRE_FRAMES = 2
FOLGA_FRAMES = 2

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

def calcular_vetores_referencia(arquivos_img, num_frames_bg, num_cortes_v, num_cortes_h):
    """Calcula a luminosidade de fundo média para os cortes verticais e horizontais."""
    print(f"FASE DE DETECÇÃO: Iniciando calibração com os primeiros {num_frames_bg} frames...")
    if len(arquivos_img) < num_frames_bg:
        raise ValueError(f"A pasta contém apenas {len(arquivos_img)} imagens.")
    
    frames_para_bg = arquivos_img[:num_frames_bg]
    soma_medias_v = np.zeros(num_cortes_v)
    soma_medias_h = np.zeros(num_cortes_h)
    
    for i, caminho_frame in enumerate(frames_para_bg):
        print(f"   Lendo frame de background {i+1}/{num_frames_bg}...", end='\r')
        img_cinza = cv2.imread(caminho_frame, cv2.IMREAD_GRAYSCALE)
        if img_cinza is None: continue
        
        cortes_v = np.array_split(img_cinza, num_cortes_v, axis=1)
        soma_medias_v += np.array([np.mean(corte) for corte in cortes_v])
        
        cortes_h = np.array_split(img_cinza, num_cortes_h, axis=0)
        soma_medias_h += np.array([np.mean(corte) for corte in cortes_h])
        
    print("\nCalibração concluída.")
    return (soma_medias_v / num_frames_bg), (soma_medias_h / num_frames_bg)

def analisar_frame(img_path, vetor_ref_v, vetor_ref_h, num_cortes_v, num_cortes_h):
    """Calcula as métricas globais e a nova métrica de contiguidade para um frame."""
    img_cinza = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cinza is None: return None
    
    # Análise Vertical
    cortes_v = np.array_split(img_cinza, num_cortes_v, axis=1)
    medias_abs_v = [np.mean(corte) for corte in cortes_v]
    lums_rel_v = np.array(medias_abs_v) - vetor_ref_v
    dp_v = np.std(lums_rel_v)
    vmax_v = np.max(lums_rel_v)

    # Cálculo da Métrica de Contiguidade Vertical
    contiguidade_v = 0
    if USAR_ANALISE_CONTIGUIDADE:
        fatias_ativas_v = lums_rel_v > LIMIAR_BRILHO_CONTIGUIDADE
        contiguidade_v = encontrar_maior_sequencia_contigua(fatias_ativas_v)

    # Análise Horizontal
    cortes_h = np.array_split(img_cinza, num_cortes_h, axis=0)
    medias_abs_h = [np.mean(corte) for corte in cortes_h]
    lums_rel_h = np.array(medias_abs_h) - vetor_ref_h
    dp_h = np.std(lums_rel_h)
    vmax_h = np.max(lums_rel_h)

    # Cálculo da Métrica de Contiguidade Horizontal
    contiguidade_h = 0
    if USAR_ANALISE_CONTIGUIDADE:
        fatias_ativas_h = lums_rel_h > LIMIAR_BRILHO_CONTIGUIDADE
        contiguidade_h = encontrar_maior_sequencia_contigua(fatias_ativas_h)
    
    return (dp_v, vmax_v, lums_rel_v, contiguidade_v), (dp_h, vmax_h, lums_rel_h, contiguidade_h)

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

def desenhar_grafico_vertical(ax, lums_rel):
    indices = np.arange(len(lums_rel))
    cores = ['#2ca02c' if val >= 0 else '#d62728' for val in lums_rel]
    ax.bar(indices, lums_rel, color=cores)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Lum. Relativa', fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.margins(x=0)

def desenhar_grafico_horizontal(ax, lums_rel):
    indices = np.arange(len(lums_rel))
    cores = ['#2ca02c' if val >= 0 else '#d62728' for val in lums_rel]
    ax.barh(indices, lums_rel, color=cores)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Lum. Relativa', fontsize=10)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.margins(y=0)
    ax.invert_yaxis()

def salvar_evento_classificado(caminho_saida, nome_pasta_evento, evento, arquivos_img, all_lums_v, all_lums_h, all_max_vals_v):
    inicio, fim = evento
    caminho_completo_evento = os.path.join(caminho_saida, nome_pasta_evento)
    os.makedirs(caminho_completo_evento, exist_ok=True)
    print(f"   -> SALVANDO Evento: {fim - inicio + 1} frames ({inicio} a {fim})")
    intervalo_max_vals = [val for i, val in enumerate(all_max_vals_v) if inicio <= i <= fim]
    if not intervalo_max_vals: return
    indice_local_pico = np.argmax(intervalo_max_vals)
    indice_global_pico = inicio + indice_local_pico
    lums_pico_v = all_lums_v[indice_global_pico]
    lums_pico_h = all_lums_h[indice_global_pico]
    caminho_img_pico = arquivos_img[indice_global_pico]
    nome_grafico = f"Analise_Pico_Frame_{indice_global_pico}.png"
    caminho_grafico = os.path.join(caminho_completo_evento, nome_grafico)
    fig = plt.figure(figsize=(16, 9))
    gs = gridspec.GridSpec(4, 5, figure=fig)
    ax_img = fig.add_subplot(gs[0:3, 0:4])
    ax_v = fig.add_subplot(gs[3, 0:4])
    ax_h = fig.add_subplot(gs[0:3, 4])
    img_rgb = cv2.cvtColor(cv2.imread(caminho_img_pico), cv2.COLOR_BGR2RGB)
    ax_img.imshow(img_rgb)
    ax_img.set_title(f'Imagem do Frame de Pico ({indice_global_pico})')
    ax_img.axis('off')
    desenhar_grafico_vertical(ax_v, lums_pico_v)
    desenhar_grafico_horizontal(ax_h, lums_pico_h)
    fig.tight_layout()
    plt.savefig(caminho_grafico)
    plt.close(fig)
    for frame_idx in range(inicio, fim + 1):
        src_path = arquivos_img[frame_idx]
        dst_path = os.path.join(caminho_completo_evento, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    LARGURA_JANELA_PIXELS, ALTURA_JANELA_PIXELS = 1700, 980
    POSICAO_X_PIXELS = (1920 - LARGURA_JANELA_PIXELS) // 2
    POSICAO_Y_PIXELS = 10

    def posicionar_janela(fig):
        try:
            canvas_manager = plt.get_current_fig_manager()
            window = canvas_manager.window
            window.geometry(f'{LARGURA_JANELA_PIXELS}x{ALTURA_JANELA_PIXELS}+{POSICAO_X_PIXELS}+{POSICAO_Y_PIXELS}')
        except Exception:
            print("\nAviso: Não foi possível posicionar a janela automaticamente.")
    
    try:
        nome_base_video = os.path.basename(os.path.normpath(CAMINHO_PASTA_ENTRADA))
        nome_pasta_final_saida = f"{nome_base_video}{SUFIXO_PASTA_SAIDA}"
        caminho_final_saida = os.path.join(PASTA_RAIZ_SAIDAS, nome_pasta_final_saida)

        print(f"Pasta de entrada: {CAMINHO_PASTA_ENTRADA}")
        print(f"Pasta de saída será criada em: {caminho_final_saida}")
        
        os.makedirs(caminho_final_saida, exist_ok=True)
        arquivos_img = sorted(glob.glob(os.path.join(CAMINHO_PASTA_ENTRADA, '*.jpg')))
        total_frames = len(arquivos_img)
        print(f"Análise iniciada. {total_frames} frames encontrados.")
        
        tempo_por_frame_ms = (TEMPO_DE_GRAVACAO_SEGUNDOS * 1000) / total_frames
        
        vetor_ref_v, vetor_ref_h = calcular_vetores_referencia(
            arquivos_img, NUM_FRAMES_BACKGROUND, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL
        )
        
        print(f"\nFASE DE DETECÇÃO: Coletando métricas de todos os {total_frames} frames...")
        all_metrics_v = []
        all_metrics_h = []
        
        for i, img_path in enumerate(arquivos_img):
            print(f"   Processando frame {i+1}/{total_frames}...", end='\r')
            result = analisar_frame(
                img_path, vetor_ref_v, vetor_ref_h, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL
            )
            if result:
                all_metrics_v.append(result[0])
                all_metrics_h.append(result[1])
        print("\nColeta de métricas concluída.")
        
        all_std_devs_v, all_max_vals_v, all_lums_v, all_contig_v = zip(*all_metrics_v)
        all_std_devs_h, all_max_vals_h, all_lums_h, all_contig_h = zip(*all_metrics_h)
        
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
        
        print(f"\nDETECÇÃO AUTOMÁTICA CONCLUÍDA: {len(eventos_candidatos)} eventos candidatos encontrados.")

        if not eventos_candidatos:
            print("Nenhum evento candidato para validar. Encerrando.")
        else:
            print("\n--- INICIANDO SESSÃO DE CLASSIFICAÇÃO ---")
            print("Use as SETAS para navegar. Pressione 'c' para CG, 'i' para IC, 'b' para BF, 'n' para NÃO, 'q' para SAIR.")
            
            eventos_salvos = 0
            user_choice = {'key': None}

            for i, evento in enumerate(eventos_candidatos):
                inicio, fim = evento
                
                intervalo_max_vals_v = [all_max_vals_v[j] for j in range(inicio, fim + 1)]
                if not intervalo_max_vals_v: continue
                
                indice_local_pico = np.argmax(intervalo_max_vals_v)
                indice_global_pico = inicio + indice_local_pico
                state = {'current_idx': indice_global_pico}
                
                fig = plt.figure(figsize=(LARGURA_JANELA_PIXELS/100, ALTURA_JANELA_PIXELS/100), dpi=100)
                gs = gridspec.GridSpec(4, 5, figure=fig)
                ax_img = fig.add_subplot(gs[0:3, 0:4])
                ax_v = fig.add_subplot(gs[3, 0:4])
                ax_h = fig.add_subplot(gs[0:3, 4])
                
                def update_display(index_to_show):
                    caminho_img = arquivos_img[index_to_show]
                    img_rgb = cv2.cvtColor(cv2.imread(caminho_img), cv2.COLOR_BGR2RGB)
                    
                    lums_rel_v = all_lums_v[index_to_show]
                    lums_rel_h = all_lums_h[index_to_show]
                    
                    ax_img.clear(); ax_v.clear(); ax_h.clear()
                    
                    ax_img.imshow(img_rgb)
                    ax_img.set_title(f'Imagem do Frame ({index_to_show})')
                    ax_img.axis('off')
                    
                    desenhar_grafico_vertical(ax_v, lums_rel_v)
                    desenhar_grafico_horizontal(ax_h, lums_rel_h)
                    
                    fig.suptitle(f"Evento {i+1}/{len(eventos_candidatos)} | Frames {inicio}-{fim} | (c=CG / i=IC / b=BF / n=NÃO / q=SAIR)", fontsize=12)
                    fig.tight_layout(rect=[0, 0.03, 1, 0.97])
                    fig.canvas.draw_idle()

                def on_key_press(event):
                    key = event.key
                    if key == 'right':
                        if state['current_idx'] < fim: state['current_idx'] += 1
                    elif key == 'left':
                        if state['current_idx'] > inicio: state['current_idx'] -= 1
                    elif key == 'up':
                        state['current_idx'] = inicio
                    elif key == 'down':
                        state['current_idx'] = fim
                    elif key in ['c', 'i', 'b', 'n', 'q']:
                        user_choice['key'] = key
                        plt.close(event.canvas.figure)
                        return
                    update_display(state['current_idx'])

                update_display(state['current_idx'])
                posicionar_janela(fig)
                fig.canvas.mpl_connect('key_press_event', on_key_press)
                plt.show(block=True)
                
                resposta = user_choice['key']

                if resposta in ['c', 'i', 'b']:
                    if resposta == 'c': classificacao = 'CG'
                    elif resposta == 'i': classificacao = 'IC'
                    else: classificacao = 'Brilho'
                    
                    print(f"Evento {i+1} CLASSIFICADO como {classificacao}.")
                    
                    frame_referencia_idx = state['current_idx']
                    nome_arquivo_ref = os.path.basename(arquivos_img[frame_referencia_idx])
                    
                    try:
                        timestamp_str = os.path.splitext(nome_arquivo_ref)[0].split('_', 1)[1]
                    except IndexError:
                        print(f"\nAVISO: Não foi possível extrair timestamp do arquivo '{nome_arquivo_ref}'. Usando nome base.")
                        timestamp_str = os.path.splitext(nome_arquivo_ref)[0]

                    num_frames_evento = fim - inicio + 1
                    duracao_evento_ms = num_frames_evento * tempo_por_frame_ms
                    
                    nome_pasta_evento = f"{frame_referencia_idx:04d} {classificacao} {timestamp_str} Dur {duracao_evento_ms:.1f}ms"
                    
                    salvar_evento_classificado(
                        caminho_final_saida, nome_pasta_evento, evento, arquivos_img, 
                        all_lums_v, all_lums_h, all_max_vals_v
                    )
                    eventos_salvos += 1
                elif resposta == 'n':
                    print(f"Evento {i+1} REJEITADO.")
                elif resposta == 'q':
                    print("Sessão de validação encerrada pelo usuário.")
                    break
        
        print("\n--- VALIDAÇÃO CONCLUÍDA ---")
        print(f"{eventos_salvos} eventos foram salvos em: {caminho_final_saida}")

    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"\nERRO: {e}")