import cv2
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

# 1. Pasta com todos os frames JPG de entrada
CAMINHO_PASTA_ENTRADA = r"D:\Videos\videos_novos_convertidos\v9.1_FNN_Y20250813H181147.590711000_UTC"
# 2. Pasta RAIZ onde todas as pastas de resultado serão criadas.n
PASTA_RAIZ_SAIDAS = r"D:\Videos\videos_classificados"

# 3. Sufixo para adicionar ao nome da pasta de saída.
SUFIXO_PASTA_SAIDA = "_classificado"

# 4. Duração total da gravação em segundos
TEMPO_DE_GRAVACAO_SEGUNDOS = 1.266247

# 5. Parâmetros da Análise
NUM_FRAMES_BACKGROUND = 75

# 6. Parâmetros de criação de dados
NUMERO_DE_CORTES_VERTICAL = 336
NUMERO_DE_CORTES_HORIZONTAL = 252 

# 7. # Sensibilidade da classificação
PERCENTIL = 75

# 8. Parâmetros de Agrupamento de Eventos
MAX_GAP_ENTRE_FRAMES = 4
FOLGA_FRAMES = 1


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
    """Calcula as métricas verticais e os vetores de luminosidade para ambas as direções."""
    img_cinza = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cinza is None: return None, None, None, None
    
    cortes_v = np.array_split(img_cinza, num_cortes_v, axis=1)
    medias_abs_v = [np.mean(corte) for corte in cortes_v]
    lums_rel_v = np.array(medias_abs_v) - vetor_ref_v
    desvio_padrao = np.std(lums_rel_v)
    valor_maximo = np.max(lums_rel_v)

    cortes_h = np.array_split(img_cinza, num_cortes_h, axis=0)
    medias_abs_h = [np.mean(corte) for corte in cortes_h]
    lums_rel_h = np.array(medias_abs_h) - vetor_ref_h
    
    return desvio_padrao, valor_maximo, lums_rel_v, lums_rel_h

def agrupar_em_eventos(indices_chave, max_gap, folga, total_frames):
    """Agrupa frames-chave consecutivos em eventos e adiciona a folga."""
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
    ax.set_ylabel('Luminosidade Relativa', fontsize=10)
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

def salvar_evento_classificado(caminho_saida, nome_pasta_evento, evento, arquivos_img, fig_para_salvar, frame_pico_idx):
    inicio, fim = evento
    caminho_completo_evento = os.path.join(caminho_saida, nome_pasta_evento)
    os.makedirs(caminho_completo_evento, exist_ok=True)
    print(f"   -> SALVANDO Evento: {fim - inicio + 1} frames ({inicio} a {fim})")
    
    nome_screenshot = f"Evento_Classificado_Frame_{frame_pico_idx}.png"
    caminho_screenshot = os.path.join(caminho_completo_evento, nome_screenshot)
    fig_para_salvar.savefig(caminho_screenshot, dpi=150, bbox_inches='tight')

    for frame_idx in range(inicio, fim + 1):
        src_path = arquivos_img[frame_idx]
        dst_path = os.path.join(caminho_completo_evento, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    LARGURA_JANELA_PIXELS, ALTURA_JANELA_PIXELS = 1800, 950
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
        # ... (código de setup, leitura de arquivos e análise de frames permanece o mesmo) ...
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
        print(f"Tempo por frame calculado: {tempo_por_frame_ms:.2f} ms")
        
        vetor_ref_v, vetor_ref_h = calcular_vetores_referencia(arquivos_img, NUM_FRAMES_BACKGROUND, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL)
        
        print(f"\nFASE DE DETECÇÃO: Coletando métricas de todos os {total_frames} frames...")
        all_std_devs, all_max_vals, all_lums_rel_vectors = [], [], []
        all_lums_rel_vectors_h = [] 
        
        for i, img_path in enumerate(arquivos_img):
            print(f"   Processando frame {i+1}/{total_frames}...", end='\r')
            dp, vmax, lums_rel_v, lums_rel_h = analisar_frame(img_path, vetor_ref_v, vetor_ref_h, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL)
            if dp is not None:
                all_std_devs.append(dp)
                all_max_vals.append(vmax)
                all_lums_rel_vectors.append(lums_rel_v)
                all_lums_rel_vectors_h.append(lums_rel_h)
        print("\nColeta de métricas concluída.")
        
        limiar_dp_adaptativo = np.percentile(all_std_devs, PERCENTIL)
        limiar_pico_adaptativo = np.percentile(all_max_vals, PERCENTIL)
        
        frames_chave_indices = []
        for i in range(len(all_std_devs)):
            if all_std_devs[i] > limiar_dp_adaptativo and all_max_vals[i] > limiar_pico_adaptativo:
                frames_chave_indices.append(i)
        
        eventos_candidatos = agrupar_em_eventos(frames_chave_indices, MAX_GAP_ENTRE_FRAMES, FOLGA_FRAMES, total_frames)
        
        print(f"\nDETECÇÃO AUTOMÁTICA CONCLUÍDA: {len(eventos_candidatos)} eventos candidatos encontrados.")

        if not eventos_candidatos:
            print("Nenhum evento candidato para validar. Encerrando.")
        else:
            print("\n--- INICIANDO SESSÃO DE CLASSIFICAÇÃO ---")
            # --- INSTRUÇÕES MODIFICADAS ---
            print("Navegação de Frames: SETAS | Edição: 'P' (Início), 'U' (Fim)")
            print("Navegação de Eventos: 'd' (Próximo), 'a' (Anterior)")
            print("Classificação: 'c' (CG), 'i' (IC), 'b' (BF), 'n' (NÃO), 'q' (SAIR)")
            
            eventos_salvos = 0
            user_choice = {'key': None}

            evento_idx = 0
            while 0 <= evento_idx < len(eventos_candidatos):
                user_choice['key'] = None
                
                evento = eventos_candidatos[evento_idx]
                inicio_sugerido, fim_sugerido = evento
                
                intervalo_max_vals = all_max_vals[inicio_sugerido:fim_sugerido+1]
                if not intervalo_max_vals:
                    evento_idx += 1
                    continue
                
                indice_local_pico = np.argmax(intervalo_max_vals)
                indice_global_pico = inicio_sugerido + indice_local_pico
                
                state = {'current_idx': indice_global_pico}
                limites_evento = {'inicio': inicio_sugerido, 'fim': fim_sugerido}

                fig = plt.figure(figsize=(LARGURA_JANELA_PIXELS/100, ALTURA_JANELA_PIXELS/100), dpi=100)
                gs = gridspec.GridSpec(4, 5, figure=fig)
                ax_img = fig.add_subplot(gs[0:3, 0:4])
                ax_v = fig.add_subplot(gs[3, 0:4])
                ax_h = fig.add_subplot(gs[0:3, 4])
                
                def update_display(index_to_show):
                    caminho_img = arquivos_img[index_to_show]
                    img_bgr = cv2.imread(caminho_img)
                    if img_bgr is None: return
                    
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    lums_rel_v = all_lums_rel_vectors[index_to_show]
                    lums_rel_h = all_lums_rel_vectors_h[index_to_show]
                    
                    ax_img.clear(); ax_v.clear(); ax_h.clear()
                    
                    ax_img.imshow(img_rgb)
                    ax_img.set_title(f'Imagem do Frame ({index_to_show})')
                    ax_img.axis('off')
                    
                    desenhar_grafico_vertical(ax_v, lums_rel_v)
                    desenhar_grafico_horizontal(ax_h, lums_rel_h)
                    
                    titulo = (f"Evento {evento_idx+1}/{len(eventos_candidatos)} | "
                              f"Editando Frames [{limites_evento['inicio']}-{limites_evento['fim']}]")
                    fig.suptitle(titulo, fontsize=12)
                    
                    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
                    
                    pos_img = ax_img.get_position()
                    pos_h = ax_h.get_position()
                    ax_h.set_position([pos_h.x0, pos_img.y0, pos_h.width, pos_img.height])
                    
                    fig.canvas.draw_idle()

                def on_key_press(event):
                    key = event.key
                    
                    if key == 'right':
                        if state['current_idx'] < total_frames - 1: state['current_idx'] += 1
                    elif key == 'left':
                        if state['current_idx'] > 0: state['current_idx'] -= 1
                    elif key == 'up':
                        state['current_idx'] = limites_evento['inicio']
                    elif key == 'down':
                        state['current_idx'] = limites_evento['fim']
                    elif key == 'p':
                        limites_evento['inicio'] = state['current_idx']
                        print(f"** Novo INÍCIO definido para o frame: {limites_evento['inicio']} **")
                    elif key == 'u':
                        limites_evento['fim'] = state['current_idx']
                        print(f"** Novo FIM definido para o frame: {limites_evento['fim']} **")
                    
                    # --- TECLAS MODIFICADAS ---
                    elif key in ['c', 'i', 'b', 'n', 'q', 'd', 'a','l']:
                        user_choice['key'] = key
                        plt.close(event.canvas.figure)
                        return
                    
                    update_display(state['current_idx'])

                update_display(state['current_idx'])
                #posicionar_janela(fig)
                fig.canvas.mpl_connect('key_press_event', on_key_press)
                plt.show(block=True)
                
                resposta = user_choice['key']

                if resposta in ['c', 'i', 'b','l']:
                    if resposta == 'c': classificacao = 'CG'
                    elif resposta == 'i': classificacao = 'IC'
                    elif resposta == 'l': classificacao = 'LCC'
                    else: classificacao = 'Brilho' 
                    
                    print(f"Evento {evento_idx+1} CLASSIFICADO como {classificacao}.")
                    
                    frame_referencia_idx = state['current_idx']
                    nome_arquivo_ref = os.path.basename(arquivos_img[frame_referencia_idx])
                    timestamp_str = os.path.splitext(nome_arquivo_ref)[0].split('_', 1)[1]
                    
                    num_frames_evento = limites_evento['fim'] - limites_evento['inicio'] + 1
                    duracao_evento_ms = num_frames_evento * tempo_por_frame_ms
                    
                    nome_pasta_evento = f"{frame_referencia_idx:04d} {classificacao} {timestamp_str} Dur {duracao_evento_ms:.1f}ms"
                    
                    evento_final_para_salvar = (limites_evento['inicio'], limites_evento['fim'])
                    salvar_evento_classificado(caminho_final_saida, nome_pasta_evento, evento_final_para_salvar, arquivos_img, fig, frame_referencia_idx)
                    eventos_salvos += 1
                    evento_idx += 1
                
                elif resposta == 'n':
                    print(f"Evento {evento_idx+1} REJEITADO.")
                    evento_idx += 1
                
                # --- LÓGICA MODIFICADA ---
                elif resposta == 'd':
                    print("Avançando para o próximo evento...")
                    evento_idx += 1
                
                # --- LÓGICA MODIFICADA ---
                elif resposta == 'a':
                    print("Retornando ao evento anterior...")
                    evento_idx -= 1
                
                elif resposta == 'q':
                    print("Sessão de validação encerrada pelo usuário.")
                    break
        
        print("\n--- VALIDAÇÃO CONCLUÍDA ---")
        print(f"{eventos_salvos} eventos foram salvos em: {caminho_final_saida}")

    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"\nERRO: {e}")