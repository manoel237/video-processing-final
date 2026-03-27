import cv2
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt

plt.style.use('dark_background')
# ==============================================================================
# --- CONFIGURAÇÃO ---
# ==============================================================================
# 1. Pasta com todos os frames JPG de entrada
CAMINHO_PASTA_ENTRADA = r"C:\Users\manoe\Downloads\Testes\v9.1_FNN_Y202501 1H010342.553768000 (20250626_~215139_UTC)"

# 2. Pasta RAIZ onde todas as pastas de resultado serão criadas.
PASTA_RAIZ_SAIDAS = r"C:\Users\manoe\Downloads\Testes\Resultados"
# 3. Sufixo para adicionar ao nome da pasta de saída.
SUFIXO_PASTA_SAIDA = "_classificado"

# 4. Duração total da gravação em segundos (pode usar decimal, ex: 30.5)
TEMPO_DE_GRAVACAO_SEGUNDOS = 1.266

# 5. Parâmetros da Análise
NUM_FRAMES_BACKGROUND = 75
NUMERO_DE_CORTES = 336

# 6. Parâmetros de Detecção
PERCENTIL = 85

# 7. Parâmetros de Agrupamento de Eventos
MAX_GAP_ENTRE_FRAMES = 6
FOLGA_FRAMES = 2


# --- Funções do Backend ---

def calcular_vetor_referencia(arquivos_img, num_frames_bg, num_cortes):
    """Calcula a luminosidade de fundo média para cada corte vertical."""
    print(f"FASE DE DETECÇÃO: Iniciando calibração com os primeiros {num_frames_bg} frames...")
    if len(arquivos_img) < num_frames_bg:
        raise ValueError(f"A pasta contém apenas {len(arquivos_img)} imagens.")
    frames_para_bg = arquivos_img[:num_frames_bg]
    soma_das_medias = np.zeros(num_cortes)
    for i, caminho_frame in enumerate(frames_para_bg):
        print(f"  Lendo frame de background {i+1}/{num_frames_bg}...", end='\r')
        img_cinza = cv2.imread(caminho_frame, cv2.IMREAD_GRAYSCALE)
        if img_cinza is None: continue
        cortes = np.array_split(img_cinza, num_cortes, axis=1)
        soma_das_medias += np.array([np.mean(corte) for corte in cortes])
    print("\nCalibração concluída.")
    return soma_das_medias / num_frames_bg

def analisar_frame(img_path, vetor_referencia, num_cortes):
    """Calcula as métricas e o vetor de luminosidade relativa de um frame."""
    img_cinza = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img_cinza is None: return None, None, None
    cortes = np.array_split(img_cinza, num_cortes, axis=1)
    medias_abs = [np.mean(corte) for corte in cortes]
    lums_rel = np.array(medias_abs) - vetor_referencia
    desvio_padrao = np.std(lums_rel)
    valor_maximo = np.max(lums_rel)
    return desvio_padrao, valor_maximo, lums_rel

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

# ALTERADO: Função simplificada para usar coordenadas de índice (0, 1, 2...)
def desenhar_grafico_no_eixo(ax, lums_rel, frame_idx):
    """Desenha o gráfico de barras em um eixo (ax) específico do Matplotlib."""
    indices = np.arange(len(lums_rel))
    
    mask_positivos = lums_rel >= 0
    indices_positivos = indices[mask_positivos]
    valores_positivos = lums_rel[mask_positivos]
    
    # Usa width=1.0 para que as barras se toquem e preencham o espaço
    ax.bar(indices_positivos, valores_positivos, color='#2ca02c', width=1.0)
    
    ax.axhline(0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Índice da Região Vertical', fontsize=10)
    ax.set_ylabel('Luminosidade Relativa', fontsize=10)
    ax.grid(axis='y', linestyle=':', alpha=0.7)
    ax.set_ylim(bottom=0)
    # Alinha as margens para que as barras comecem e terminem nas bordas do eixo
    ax.margins(x=0)

def salvar_evento_classificado(caminho_saida, nome_pasta_evento, evento, arquivos_img, all_lums_rel, all_max_vals):
    """Cria a pasta, copia os frames e salva o gráfico para um evento classificado."""
    inicio, fim = evento
    
    caminho_completo_evento = os.path.join(caminho_saida, nome_pasta_evento)
    os.makedirs(caminho_completo_evento, exist_ok=True)
    
    print(f"  -> SALVANDO Evento: {fim - inicio + 1} frames ({inicio} a {fim})")
    
    intervalo_max_vals = all_max_vals[inicio:fim+1]
    if not intervalo_max_vals: return

    indice_local_pico = np.argmax(intervalo_max_vals)
    indice_global_pico = inicio + indice_local_pico
    
    vetor_lums_pico = all_lums_rel[indice_global_pico]
    nome_grafico = f"Grafico_Pico_Frame_{indice_global_pico}.png"
    caminho_grafico = os.path.join(caminho_completo_evento, nome_grafico)
    
    fig, ax = plt.subplots(figsize=(16, 7))
    # A função simplificada agora funciona perfeitamente aqui também
    desenhar_grafico_no_eixo(ax, vetor_lums_pico, indice_global_pico)
    plt.savefig(caminho_grafico)
    plt.close(fig)

    for frame_idx in range(inicio, fim + 1):
        src_path = arquivos_img[frame_idx]
        dst_path = os.path.join(caminho_completo_evento, os.path.basename(src_path))
        shutil.copy(src_path, dst_path)

# --- Bloco de Execução Principal ---
if __name__ == "__main__":

    # Parâmetros para controle da janela
    LARGURA_JANELA_PIXELS = 1700
    ALTURA_JANELA_PIXELS = 1020
    POSICAO_X_PIXELS = (1920 - LARGURA_JANELA_PIXELS) // 2
    POSICAO_Y_PIXELS = (1080 - ALTURA_JANELA_PIXELS) // 2

    def posicionar_janela(fig):
        try:
            canvas_manager = plt.get_current_fig_manager()
            window = canvas_manager.window
            window.geometry(f'{LARGURA_JANELA_PIXELS}x{ALTURA_JANELA_PIXELS}+{POSICAO_X_PIXELS}+{POSICAO_Y_PIXELS}')
        except Exception:
            print("\nAviso: Não foi possível posicionar a janela automaticamente.")
    
    try:
        # --- PREPARAÇÃO ---
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
        
        vetor_zero = calcular_vetor_referencia(arquivos_img, NUM_FRAMES_BACKGROUND, NUMERO_DE_CORTES)
        
        print(f"\nFASE DE DETECÇÃO: Coletando métricas de todos os {total_frames} frames...")
        all_std_devs, all_max_vals, all_lums_rel_vectors = [], [], []
        for i, img_path in enumerate(arquivos_img):
            print(f"  Processando frame {i+1}/{total_frames}...", end='\r')
            dp, vmax, lums_rel = analisar_frame(img_path, vetor_zero, NUMERO_DE_CORTES)
            if dp is not None:
                all_std_devs.append(dp)
                all_max_vals.append(vmax)
                all_lums_rel_vectors.append(lums_rel)
        print("\nColeta de métricas concluída.")
        
        limiar_dp_adaptativo = np.percentile(all_std_devs, PERCENTIL)
        limiar_pico_adaptativo = np.percentile(all_max_vals, PERCENTIL)
        
        frames_chave_indices = []
        for i in range(len(all_std_devs)):
            if all_std_devs[i] > limiar_dp_adaptativo and all_max_vals[i] > limiar_pico_adaptativo:
                frames_chave_indices.append(i)
        
        eventos_candidatos = agrupar_em_eventos(frames_chave_indices, MAX_GAP_ENTRE_FRAMES, FOLGA_FRAMES, total_frames)
        
        print(f"\nDETECÇÃO AUTOMÁTICA CONCLUÍDA: {len(eventos_candidatos)} eventos candidatos encontrados.")

        # --- FASE 2: VALIDAÇÃO E CLASSIFICAÇÃO HUMANA ---
        if not eventos_candidatos:
            print("Nenhum evento candidato para validar. Encerrando.")
        else:
            print("\n--- INICIANDO SESSÃO DE CLASSIFICAÇÃO ---")
            print("Use as SETAS (<- e ->) para navegar. Pressione 'c' para CG, 'i' para IC, 'b' para BF, 'n' para NÃO, 'q' para SAIR.")
            
            eventos_salvos = 0
            user_choice = {'key': None}

            for i, evento in enumerate(eventos_candidatos):
                inicio, fim = evento
                
                intervalo_max_vals = all_max_vals[inicio:fim+1]
                if not intervalo_max_vals: continue
                
                indice_local_pico = np.argmax(intervalo_max_vals)
                indice_global_pico = inicio + indice_local_pico
                
                state = {'current_idx': indice_global_pico}
                
                DPI = 100
                largura_polegadas = LARGURA_JANELA_PIXELS / DPI
                altura_polegadas = ALTURA_JANELA_PIXELS / DPI
                fig, axs = plt.subplots(2, 1, figsize=(largura_polegadas, altura_polegadas), gridspec_kw={'height_ratios': [7, 3]}, dpi=DPI)
                
                # ALTERADO: A função de update AGORA implementa o alinhamento de forma robusta
                def update_display(index_to_show):
                    caminho_img = arquivos_img[index_to_show]
                    img_bgr = cv2.imread(caminho_img)
                    if img_bgr is None: return
                    
                    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
                    lums_rel = all_lums_rel_vectors[index_to_show]
                    
                    axs[0].clear()
                    axs[1].clear()
                    
                    # Desenha a imagem no eixo de cima
                    axs[0].imshow(img_rgb)
                    axs[0].set_title(f'Imagem do Frame ({index_to_show})')
                    axs[0].axis('off')
                    
                    # Desenha o gráfico no eixo de baixo
                    desenhar_grafico_no_eixo(axs[1], lums_rel, index_to_show)
                    
                    fig.suptitle(f"Evento {i+1}/{len(eventos_candidatos)} | Frames {inicio}-{fim} | (c=CG / i=IC / b=BF / n=NÃO / q=SAIR)", fontsize=12)
                    
                    # NOVO: Lógica de alinhamento pós-desenho
                    fig.canvas.draw() # Força a renderização
                    
                    # Pega a posição exata da imagem e força o gráfico a usar a mesma
                    pos_imagem = axs[0].get_position()
                    pos_grafico = axs[1].get_position()
                    axs[1].set_position([pos_imagem.x0, pos_grafico.y0, pos_imagem.width, pos_grafico.height])
                    
                    fig.canvas.draw_idle()

                def on_key_press(event):
                    key = event.key
                    
                    if key == 'right':
                        if state['current_idx'] < fim:
                            state['current_idx'] += 1
                            update_display(state['current_idx'])
                    elif key == 'left':
                        if state['current_idx'] > inicio:
                            state['current_idx'] -= 1
                            update_display(state['current_idx'])
                    elif key in ['c', 'i', 'b', 'n', 'q']:
                        user_choice['key'] = key
                        plt.close(event.canvas.figure)

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
                    timestamp_str = os.path.splitext(nome_arquivo_ref)[0].split('_', 1)[1]

                    num_frames_evento = fim - inicio + 1
                    duracao_evento_ms = num_frames_evento * tempo_por_frame_ms
                    
                    nome_pasta_evento = f"{frame_referencia_idx:04d} {classificacao} {timestamp_str} Dur {duracao_evento_ms:.1f}ms"
                    
                    salvar_evento_classificado(caminho_final_saida, nome_pasta_evento, evento, arquivos_img, all_lums_rel_vectors, all_max_vals)
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