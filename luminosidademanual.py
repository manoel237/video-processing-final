import cv2
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.style.use('dark_background')

# 1. Pasta com todos os frames JPG de entrada
CAMINHO_PASTA_ENTRADA = r"C:\Arquivos\Videos\Phantom\v9.1_FNN_Y20250721H104750.975719000_UTC"

# 2. Pasta RAIZ onde todas as pastas de resultado serão criadas.
PASTA_RAIZ_SAIDAS = r"C:\Arquivos\Videos\videos_classificados"
# 3. Sufixo para adicionar ao nome da pasta de saída.
SUFIXO_PASTA_SAIDA = "_classificado"

# 4. Duração total da gravação em segundos
TEMPO_DE_GRAVACAO_SEGUNDOS = 1.266247

# 5. PARÂMETROS DA ANÁLISE MANUAL
FRAME_CENTRAL_ANALISE = 653
RAIO_FRAMES = 20

# 6. Parâmetros da Análise de Fundo
NUM_FRAMES_BACKGROUND = 75
NUMERO_DE_CORTES_VERTICAL = 336
NUMERO_DE_CORTES_HORIZONTAL = 252 

def calcular_vetores_referencia(arquivos_img, num_frames_bg, num_cortes_v, num_cortes_h):
    """Calcula a luminosidade de fundo média para os cortes verticais e horizontais."""
    print(f"FASE DE CALIBRAÇÃO: Iniciando com os primeiros {num_frames_bg} frames...")
    if len(arquivos_img) < num_frames_bg:
        raise ValueError(f"A pasta contém apenas {len(arquivos_img)} imagens, mas são necessárias {num_frames_bg} para a calibração.")
    
    frames_para_bg = arquivos_img[:num_frames_bg]
    soma_medias_v = np.zeros(num_cortes_v)
    soma_medias_h = np.zeros(num_cortes_h) 
    
    for i, caminho_frame in enumerate(frames_para_bg):
        print(f"   Lendo frame de background {i+1}/{num_frames_bg}...", end='\r')
        img_cinza = cv2.imread(caminho_frame, cv2.IMREAD_GRAYSCALE)
        if img_cinza is None: continue
        
        # Análise Vertical
        cortes_v = np.array_split(img_cinza, num_cortes_v, axis=1)
        soma_medias_v += np.array([np.mean(corte) for corte in cortes_v])
        
        # Análise Horizontal
        cortes_h = np.array_split(img_cinza, num_cortes_h, axis=0)
        soma_medias_h += np.array([np.mean(corte) for corte in cortes_h])

    print("\nCalibração concluída.")
    return (soma_medias_v / num_frames_bg), (soma_medias_h / num_frames_bg)

def analisar_frame(img_path, vetor_ref_v, vetor_ref_h, num_cortes_v, num_cortes_h):
    """Calcula as métricas e os vetores de luminosidade relativa de um frame."""
    img_cinza = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cinza is None: return None, None, None
    
    # Análise Vertical
    cortes_v = np.array_split(img_cinza, num_cortes_v, axis=1)
    medias_abs_v = [np.mean(corte) for corte in cortes_v]
    lums_rel_v = np.array(medias_abs_v) - vetor_ref_v
    valor_maximo = np.max(lums_rel_v)

    # Análise Horizontal
    cortes_h = np.array_split(img_cinza, num_cortes_h, axis=0)
    medias_abs_h = [np.mean(corte) for corte in cortes_h]
    lums_rel_h = np.array(medias_abs_h) - vetor_ref_h
    
    return valor_maximo, lums_rel_v, lums_rel_h

def desenhar_grafico_vertical(ax, lums_rel):
    """Desenha o gráfico de barras verticais em um eixo."""
    indices = np.arange(len(lums_rel))
    cores = ['#2ca02c' if val >= 0 else '#d62728' for val in lums_rel]
    ax.bar(indices, lums_rel, color=cores)
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_ylabel('Luminosidade Relativa', fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.margins(x=0)

def desenhar_grafico_horizontal(ax, lums_rel):
    """Desenha o gráfico de barras horizontais em um eixo."""
    indices = np.arange(len(lums_rel))
    cores = ['#2ca02c' if val >= 0 else '#d62728' for val in lums_rel]
    ax.barh(indices, lums_rel, color=cores)
    ax.axvline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Lum. Relativa', fontsize=10)
    ax.grid(axis='x', linestyle=':', alpha=0.7)
    ax.margins(y=0)
    ax.invert_yaxis()

# --- FUNÇÃO MODIFICADA ---
def salvar_evento_classificado(caminho_saida, nome_pasta_evento, evento, arquivos_img, fig_para_salvar, frame_ref_idx):
    """Cria a pasta, copia os frames e salva a janela de visualização completa."""
    inicio, fim = evento
    caminho_completo_evento = os.path.join(caminho_saida, nome_pasta_evento)
    os.makedirs(caminho_completo_evento, exist_ok=True)
    print(f"   -> SALVANDO Evento: {fim - inicio + 1} frames ({inicio} a {fim})")
    
    # Salva a figura COMPLETA que foi passada como argumento
    nome_screenshot = f"Analise_Visual_Frame_{frame_ref_idx}.png"
    caminho_screenshot = os.path.join(caminho_completo_evento, nome_screenshot)
    fig_para_salvar.savefig(caminho_screenshot, dpi=150, bbox_inches='tight')

    # Copia os frames do intervalo
    for frame_idx in range(inicio, fim + 1):
        src_path = arquivos_img[frame_idx]
        dst_path = os.path.join(caminho_completo_evento, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

# --- Bloco de Execução Principal ---
if __name__ == "__main__":
    LARGURA_JANELA_PIXELS = 1700
    ALTURA_JANELA_PIXELS = 980
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
        print(f"Tempo por frame calculado: {tempo_por_frame_ms:.2f} ms")
        
        vetor_ref_v, vetor_ref_h = calcular_vetores_referencia(arquivos_img, NUM_FRAMES_BACKGROUND, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL)
        
        print(f"\nFASE DE ANÁLISE: Coletando métricas de todos os {total_frames} frames...")
        all_max_vals, all_lums_rel_vectors_v, all_lums_rel_vectors_h = [], [], []
        for i, img_path in enumerate(arquivos_img):
            print(f"   Processando frame {i+1}/{total_frames}...", end='\r')
            vmax, lums_rel_v, lums_rel_h = analisar_frame(img_path, vetor_ref_v, vetor_ref_h, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL)
            if vmax is not None:
                all_max_vals.append(vmax)
                all_lums_rel_vectors_v.append(lums_rel_v)
                all_lums_rel_vectors_h.append(lums_rel_h)
        print("\nColeta de métricas concluída.")
        
        inicio_intervalo = max(0, FRAME_CENTRAL_ANALISE - RAIO_FRAMES)
        fim_intervalo = min(total_frames - 1, FRAME_CENTRAL_ANALISE + RAIO_FRAMES)
        if FRAME_CENTRAL_ANALISE >= total_frames:
            raise ValueError(f"O FRAME_CENTRAL_ANALISE ({FRAME_CENTRAL_ANALISE}) é maior ou igual ao total de frames ({total_frames}).")

        print("\n--- INICIANDO SESSÃO DE ANÁLISE MANUAL ---")
        print(f"Analisando o intervalo de frames: {inicio_intervalo} a {fim_intervalo}")
        print("Use as SETAS (<-,->, para cima e para baixo) para navegar. Pressione 'c' para CG, 'i' para IC, 'b' para BF, 'n' para NÃO, 'q' para SAIR.")
        
        user_choice = {'key': None}
        state = {'current_idx': FRAME_CENTRAL_ANALISE}
        evento_analisado = (inicio_intervalo, fim_intervalo)

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
            lums_rel_v = all_lums_rel_vectors_v[index_to_show]
            lums_rel_h = all_lums_rel_vectors_h[index_to_show] 
            
            ax_img.clear(); ax_v.clear(); ax_h.clear()
            
            ax_img.imshow(img_rgb)
            ax_img.set_title(f'Imagem do Frame ({index_to_show})')
            ax_img.axis('off')
            
            desenhar_grafico_vertical(ax_v, lums_rel_v)
            desenhar_grafico_horizontal(ax_h, lums_rel_h)
            
            fig.suptitle(f"Análise Manual | Frames {inicio_intervalo}-{fim_intervalo} | (c=CG / i=IC / b=BF / n=NÃO / q=SAIR)", fontsize=12)
            fig.tight_layout(rect=[0, 0.03, 1, 0.97])
            fig.canvas.draw_idle()

        def on_key_press(event):
            key = event.key
            if key == 'right':
                if state['current_idx'] < fim_intervalo: 
                    state['current_idx'] += 1
                    update_display(state['current_idx'])
            elif key == 'left':
                if state['current_idx'] > inicio_intervalo: 
                    state['current_idx'] -= 1
                    update_display(state['current_idx'])
            elif key == 'up':
                if state['current_idx'] != inicio_intervalo: 
                    state['current_idx'] = inicio_intervalo
                    update_display(state['current_idx'])
            elif key == 'down':
                if state['current_idx'] != fim_intervalo: 
                    state['current_idx'] = fim_intervalo
                    update_display(state['current_idx'])
            elif key in ['c', 'i', 'b', 'n', 'q']:
                user_choice['key'] = key
                # Não fecha a figura aqui, para que possamos salvá-la
                plt.close(event.canvas.figure)
                return
            
        update_display(state['current_idx'])
        posicionar_janela(fig)
        fig.canvas.mpl_connect('key_press_event', on_key_press)
        plt.show(block=True)
        
        resposta = user_choice['key']
        
        if resposta in ['c', 'i', 'b']:
            if resposta == 'c': classificacao = 'CG'
            elif resposta == 'i': classificacao = 'IC'
            else: classificacao = 'Brilho'
            
            print(f"Intervalo CLASSIFICADO como {classificacao}.")
            
            frame_referencia_idx = state['current_idx']
            nome_arquivo_ref = os.path.basename(arquivos_img[frame_referencia_idx])
            timestamp_str = os.path.splitext(nome_arquivo_ref)[0].split('_', 1)[1]
            num_frames_evento = fim_intervalo - inicio_intervalo + 1
            duracao_evento_ms = num_frames_evento * tempo_por_frame_ms
            nome_pasta_evento = f"{frame_referencia_idx:04d} {classificacao} {timestamp_str} Dur {duracao_evento_ms:.1f}ms"
            
            # --- CHAMADA MODIFICADA ---
            salvar_evento_classificado(caminho_final_saida, nome_pasta_evento, evento_analisado, arquivos_img, fig, frame_referencia_idx)
            print(f"Evento salvo em: {os.path.join(caminho_final_saida, nome_pasta_evento)}")
            
        elif resposta == 'n':
            print("Intervalo REJEITADO.")
        elif resposta == 'q':
            print("Sessão de análise encerrada pelo usuário.")
    
    except (FileNotFoundError, ValueError, IndexError) as e:
        print(f"\nERRO: {e}")