import cv2
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import sys
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- CONFIGURAÇÕES DE TUNING ---
# Ajuste conforme sua máquina. None = Usa todos os núcleos disponíveis.
NUM_PROCESSOS_PARALELOS = None 

plt.style.use('dark_background')

# --- CONFIGURAÇÕES DO PROJETO ---
PASTA_VIDEOS_NOVOS = r"D:\videos_novos\manoel\manoel\convertidos"
PASTA_RAIZ_SAIDAS = r"D:\videos_novos\manoel\manoel\convertidos\out"
SUFIXO_PASTA_SAIDA = "_classificado"

# Parâmetros Físicos
TEMPO_DE_GRAVACAO_SEGUNDOS = 1.266247
NUM_FRAMES_BACKGROUND = 75
NUMERO_DE_CORTES_VERTICAL = 336
NUMERO_DE_CORTES_HORIZONTAL = 252 
PERCENTIL = 75
MAX_GAP_ENTRE_FRAMES = 4
FOLGA_FRAMES = 1

# --- WORKER DE PROCESSAMENTO (PARALELO) ---
def worker_analisar_frame(args):
    """
    Função isolada que roda em núcleos separados.
    Recebe todos os dados necessários empacotados em 'args'.
    """
    img_path, vetor_ref_v, vetor_ref_h, n_v, n_h = args
    
    # Leitura otimizada (flag 0 = grayscale direto)
    img_cinza = cv2.imread(img_path, 0) 
    if img_cinza is None: 
        return (None, None, None, None)
    
    # Processamento Vetorizado
    # Dividir verticalmente
    cortes_v = np.array_split(img_cinza, n_v, axis=1)
    # A média de cada corte - vetor de referência
    lums_rel_v = np.array([np.mean(c) for c in cortes_v]) - vetor_ref_v
    
    # Dividir horizontalmente
    cortes_h = np.array_split(img_cinza, n_h, axis=0)
    lums_rel_h = np.array([np.mean(c) for c in cortes_h]) - vetor_ref_h
    
    # Cálculo estatístico leve
    std_dev = np.std(lums_rel_v)
    max_val = np.max(lums_rel_v)
    
    return (std_dev, max_val, lums_rel_v, lums_rel_h)

# --- CLASSE DA INTERFACE GRÁFICA ---
class AppAnaliseRaios:
    def __init__(self, root):
        self.root = root
        self.root.title("⚡ DETECTOR DE RAIOS - TURBO SUPER TUNADO 5000 ⚡")
        self.root.geometry("1000x700")
        self.root.configure(bg="#101010")
        
        # --- ESTILIZAÇÃO (Tema Hacker/Pro) ---
        style = ttk.Style()
        style.theme_use('clam')
        
        # Cores da Treeview
        style.configure("Treeview", 
                        background="#202020", 
                        foreground="#00ff00", 
                        fieldbackground="#202020", 
                        font=('Consolas', 10))
        
        style.configure("Treeview.Heading", 
                        background="#333", 
                        foreground="white", 
                        font=('Arial', 10, 'bold'))
        
        style.map('Treeview', background=[('selected', '#004400')])

        # Layout Principal (PanedWindow para redimensionar)
        self.paned = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#101010", sashwidth=4)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # --- PAINEL ESQUERDO (LISTA) ---
        frame_left = tk.Frame(self.paned, bg="#101010")
        self.paned.add(frame_left, minsize=400)
        
        lbl_list = tk.Label(frame_left, text="LISTA DE VÍDEOS", bg="#101010", fg="#00ff00", font=("Impact", 12))
        lbl_list.pack(pady=5)
        
        # Treeview
        self.tree = ttk.Treeview(frame_left, columns=("num", "nome", "status"), show='headings', selectmode='browse')
        self.tree.heading("num", text="#")
        self.tree.heading("nome", text="Arquivo")
        self.tree.heading("status", text="Estado")
        
        self.tree.column("num", width=40, anchor='center')
        self.tree.column("nome", width=300, anchor='w')
        self.tree.column("status", width=80, anchor='center')
        
        # Tag para linha verde
        self.tree.tag_configure('concluido', background='#003300', foreground='#00ff00')
        self.tree.tag_configure('pendente', foreground='white')
        
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # Scrollbar da lista
        scroll_tree = ttk.Scrollbar(frame_left, orient="vertical", command=self.tree.yview)
        scroll_tree.place(relx=0.96, rely=0.05, relheight=0.95)
        self.tree.configure(yscroll=scroll_tree.set)

        # --- PAINEL DIREITO (LOGS & CONTROLE) ---
        frame_right = tk.Frame(self.paned, bg="#151515")
        self.paned.add(frame_right, minsize=400)
        
        lbl_log = tk.Label(frame_right, text="LOG DE SISTEMA", bg="#151515", fg="#00ff00", font=("Impact", 12))
        lbl_log.pack(pady=5)
        
        # Log Text Area
        self.log_text = scrolledtext.ScrolledText(frame_right, bg="black", fg="#00ff00", font=("Consolas", 9), state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5)
        
        # Área de Botões
        frame_btns = tk.Frame(frame_right, bg="#151515")
        frame_btns.pack(pady=10, fill=tk.X)
        
        btn_refresh = tk.Button(frame_btns, text="ATUALIZAR LISTA", command=self.carregar_lista, 
                                bg="#333", fg="white", font=("Arial", 10, "bold"), relief="flat", padx=10, pady=5)
        btn_refresh.pack(side=tk.LEFT, padx=10)
        
        btn_start = tk.Button(frame_btns, text="▶ INICIAR ANÁLISE", command=self.iniciar_processamento, 
                              bg="#006600", fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=5)
        btn_start.pack(side=tk.RIGHT, padx=10)

        # Dados internos
        self.mapa_caminhos = {}
        self.carregar_lista()
        
        self.log("Sistema Iniciado. Pronto para processamento paralelo.")
        cpu_count = multiprocessing.cpu_count()
        self.log(f"Núcleos de CPU detectados: {cpu_count}")

    def log(self, msg):
        """Escreve no painel de log e rola para baixo."""
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f">> {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.root.update() # Força atualização da GUI

    def carregar_lista(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.mapa_caminhos = {}
        
        if not os.path.exists(PASTA_VIDEOS_NOVOS):
            self.log(f"ERRO: Pasta não encontrada: {PASTA_VIDEOS_NOVOS}")
            return

        pastas = sorted([d for d in os.listdir(PASTA_VIDEOS_NOVOS) if os.path.isdir(os.path.join(PASTA_VIDEOS_NOVOS, d))])
        
        for i, nome_pasta in enumerate(pastas):
            iid = self.tree.insert("", "end", values=(f"{i+1:02d}", nome_pasta, "Pendente"), tags=('pendente',))
            self.mapa_caminhos[iid] = os.path.join(PASTA_VIDEOS_NOVOS, nome_pasta)
            
        self.log(f"Lista atualizada: {len(pastas)} vídeos encontrados.")

    def iniciar_processamento(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Aviso", "Selecione um vídeo na lista!")
            return
        
        item_id = selected[0]
        caminho_pasta = self.mapa_caminhos[item_id]
        
        # GUI: Muda status para "Processando..."
        self.tree.set(item_id, "status", "Rodando...")
        self.root.config(cursor="watch")
        
        try:
            sucesso = self.executar_analise_video(caminho_pasta)
            
            if sucesso:
                self.tree.set(item_id, "status", "Concluído")
                self.tree.item(item_id, tags=('concluido',))
                self.log(f"Sucesso. Vídeo marcado como concluído.")
            else:
                self.tree.set(item_id, "status", "Vazio/Erro")
                
        except Exception as e:
            self.log(f"ERRO CRÍTICO: {e}")
            messagebox.showerror("Erro", str(e))
        finally:
            self.root.config(cursor="")

    # --- LÓGICA PESADA DE ANÁLISE ---
    def executar_analise_video(self, caminho_pasta_entrada):
        nome_base = os.path.basename(os.path.normpath(caminho_pasta_entrada))
        self.log(f"--- INICIANDO: {nome_base} ---")
        
        arquivos_img = sorted(glob.glob(os.path.join(caminho_pasta_entrada, '*.jpg')))
        total_frames = len(arquivos_img)
        
        if total_frames < NUM_FRAMES_BACKGROUND:
            self.log("Vídeo muito curto para análise.")
            return False

        # 1. Calibração (Rápida, single thread é suficiente)
        self.log("Calibrando background...")
        vetor_v, vetor_h = self.calcular_vetores_referencia(arquivos_img)
        
        # 2. Processamento Paralelo (AQUI ESTÁ A OTIMIZAÇÃO DE VELOCIDADE)
        self.log(f"Iniciando Scan em Paralelo ({total_frames} frames)...")
        t_inicio = time.time()
        
        # Prepara argumentos para os workers
        # Lista de tuplas: (path, vet_v, vet_h, n_v, n_h)
        args_list = [
            (f, vetor_v, vetor_h, NUMERO_DE_CORTES_VERTICAL, NUMERO_DE_CORTES_HORIZONTAL) 
            for f in arquivos_img
        ]
        
        dados_processados = []
        with ProcessPoolExecutor(max_workers=NUM_PROCESSOS_PARALELOS) as executor:
            # map mantém a ordem, importante para vídeo
            resultados = executor.map(worker_analisar_frame, args_list)
            
            # Coleta resultados mantendo a GUI viva
            for i, res in enumerate(resultados):
                dados_processados.append(res)
                if i % 100 == 0: # Atualiza log a cada 100 frames
                    progresso = (i / total_frames) * 100
        
        tempo_total = time.time() - t_inicio
        fps_proc = total_frames / tempo_total
        self.log(f"Scan concluído em {tempo_total:.2f}s ({fps_proc:.1f} FPS).")

        # Separar dados
        all_std = []
        all_max = []
        valid_indices = []
        
        # Filtrar frames ruins (None)
        final_dados = [] # Guarda tuplas completas para usar na plotagem
        
        for i, (dp, mx, vv, vh) in enumerate(dados_processados):
            if dp is not None:
                all_std.append(dp)
                all_max.append(mx)
                final_dados.append((vv, vh)) # Guarda vetores para plotar rápido depois
                valid_indices.append(i)
            else:
                # Mantem placeholder para manter indice alinhado com arquivos_img
                all_std.append(0)
                all_max.append(0)
                final_dados.append((np.zeros(NUMERO_DE_CORTES_VERTICAL), np.zeros(NUMERO_DE_CORTES_HORIZONTAL)))

        # 3. Detecção de Eventos
        limiar_std = np.percentile(all_std, PERCENTIL)
        limiar_max = np.percentile(all_max, PERCENTIL)
        
        indices_chave = [i for i in range(len(all_std)) if all_std[i] > limiar_std and all_max[i] > limiar_max]
        eventos = self.agrupar_em_eventos(indices_chave, total_frames)
        
        self.log(f"Eventos candidatos encontrados: {len(eventos)}")
        
        if not eventos:
            return False

        # 4. Interface de Classificação Otimizada
        self.abrir_editor_visual(eventos, arquivos_img, all_max, final_dados, nome_base, caminho_pasta_entrada)
        return True

    def calcular_vetores_referencia(self, arquivos_img):
        # Apenas os primeiros N frames
        frames = arquivos_img[:NUM_FRAMES_BACKGROUND]
        soma_v = np.zeros(NUMERO_DE_CORTES_VERTICAL)
        soma_h = np.zeros(NUMERO_DE_CORTES_HORIZONTAL)
        count = 0
        
        for p in frames:
            img = cv2.imread(p, 0)
            if img is None: continue
            
            parts_v = np.array_split(img, NUMERO_DE_CORTES_VERTICAL, axis=1)
            soma_v += np.array([np.mean(c) for c in parts_v])
            
            parts_h = np.array_split(img, NUMERO_DE_CORTES_HORIZONTAL, axis=0)
            soma_h += np.array([np.mean(c) for c in parts_h])
            count += 1
            
        return (soma_v / count), (soma_h / count)

    def agrupar_em_eventos(self, indices, total):
        if not indices: return []
        evs = []
        atual = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] - atual[-1] <= MAX_GAP_ENTRE_FRAMES + 1:
                atual.append(indices[i])
            else:
                ini = max(0, min(atual) - FOLGA_FRAMES)
                fim = min(total - 1, max(atual) + FOLGA_FRAMES)
                evs.append((ini, fim))
                atual = [indices[i]]
        ini = max(0, min(atual) - FOLGA_FRAMES)
        fim = min(total - 1, max(atual) + FOLGA_FRAMES)
        evs.append((ini, fim))
        return evs

    # --- EDITOR VISUAL OTIMIZADO (MAXIMIZADO AUTOMATICAMENTE + ESCALA DINÂMICA) ---
    def abrir_editor_visual(self, eventos, arquivos_img, all_max, dados_vetores, nome_video, pasta_origem):
        self.log("Abrindo Editor Visual...")
        
        # Setup de Pastas
        pasta_saida_final = os.path.join(PASTA_RAIZ_SAIDAS, f"{nome_video}{SUFIXO_PASTA_SAIDA}")
        os.makedirs(pasta_saida_final, exist_ok=True)
        
        # Variáveis de Controle
        state = {
            'evt_idx': 0,
            'frame_idx': 0,
            'inicio': 0,
            'fim': 0,
            'salvos': 0,
            'running': True
        }

        # Inicializa Figura
        fig = plt.figure(figsize=(16, 9))
        
        # --- FORÇA A JANELA A ABRIR MAXIMIZADA ---
        try:
            manager = plt.get_current_fig_manager()
            manager.window.state('zoomed') 
        except Exception:
            pass 
        # ----------------------------------------

        gs = gridspec.GridSpec(4, 5, figure=fig)
        ax_img = fig.add_subplot(gs[0:3, 0:4])
        ax_v = fig.add_subplot(gs[3, 0:4])
        ax_h = fig.add_subplot(gs[0:3, 4])
        
        # Ajusta margens
        plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.1, hspace=0.1)

        # Objetos Gráficos Persistentes
        img_placeholder = np.zeros((100, 100, 3), dtype=np.uint8)
        im_obj = ax_img.imshow(img_placeholder)
        ax_img.axis('off')
        
        # Barras iniciais
        bar_v = ax_v.bar(np.arange(NUMERO_DE_CORTES_VERTICAL), np.zeros(NUMERO_DE_CORTES_VERTICAL))
        ax_v.grid(axis='y', linestyle=':', alpha=0.5)
        
        bar_h = ax_h.barh(np.arange(NUMERO_DE_CORTES_HORIZONTAL), np.zeros(NUMERO_DE_CORTES_HORIZONTAL))
        ax_h.invert_yaxis()
        ax_h.grid(axis='x', linestyle=':', alpha=0.5)
        
        texto_titulo = fig.suptitle("Inicializando...", fontsize=14, color='white')

        def atualizar_evento():
            if not (0 <= state['evt_idx'] < len(eventos)):
                state['running'] = False
                plt.close(fig)
                return

            inicio, fim = eventos[state['evt_idx']]
            sub_max = all_max[inicio:fim+1]
            if not sub_max:
                state['evt_idx'] += 1
                atualizar_evento()
                return

            pico_local = np.argmax(sub_max)
            state['frame_idx'] = inicio + pico_local
            state['inicio'] = inicio
            state['fim'] = fim
            renderizar_frame()

        def renderizar_frame():
            idx = state['frame_idx']
            
            # 1. Atualizar Imagem
            bgr = cv2.imread(arquivos_img[idx])
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                im_obj.set_data(rgb)
                im_obj.set_extent((0, rgb.shape[1], rgb.shape[0], 0))
            
            # 2. Atualizar Gráficos
            v_v, v_h = dados_vetores[idx]
            
            # --- ESCALA DINÂMICA INTELIGENTE VERTICAL ---
            peak_v = np.max(np.abs(v_v))
            # O limite é o pico do sinal + 10% de folga.
            # Mínimo de 1.0 para não quebrar a escala em frames pretos.
            limite_v = max(peak_v * 1.1, 1.0) 
            ax_v.set_ylim(-limite_v, limite_v)
            
            for rect, val in zip(bar_v, v_v):
                rect.set_height(val)
                rect.set_color('#00ff00' if val >= 0 else '#ff0000')

            # --- ESCALA DINÂMICA INTELIGENTE HORIZONTAL ---
            peak_h = np.max(np.abs(v_h))
            limite_h = max(peak_h * 1.1, 1.0)
            ax_h.set_xlim(-limite_h, limite_h)
            
            for rect, val in zip(bar_h, v_h):
                rect.set_width(val)
                rect.set_color('#00ff00' if val >= 0 else '#ff0000')

            # 3. Atualizar Título
            evt_display = f"EVENTO {state['evt_idx']+1}/{len(eventos)}"
            frame_display = f"Frame: {idx} | Range: {state['inicio']}-{state['fim']}"
            texto_titulo.set_text(f"{evt_display} | {frame_display}\nCOMANDOS: Setas(Nav) | P(Start)/U(End) | C/I/L/B(Classificar) | N(Pular)")
            
            fig.canvas.draw_idle()

        def salvar_selecao(tipo):
            dur_ms = (state['fim'] - state['inicio'] + 1) * ((TEMPO_DE_GRAVACAO_SEGUNDOS * 1000) / len(arquivos_img))
            ref_idx = state['frame_idx']
            ts = os.path.splitext(os.path.basename(arquivos_img[ref_idx]))[0].split('_', 1)[1]
            
            nome_pasta = f"{ref_idx:04d} {tipo} {ts} Dur {dur_ms:.1f}ms"
            path_evt = os.path.join(pasta_saida_final, nome_pasta)
            os.makedirs(path_evt, exist_ok=True)
            
            # Salvar print
            fig.savefig(os.path.join(path_evt, f"Info_Frame_{ref_idx}.png"), bbox_inches='tight')
            
            # Copiar frames
            for k in range(state['inicio'], state['fim'] + 1):
                shutil.copy(arquivos_img[k], os.path.join(path_evt, os.path.basename(arquivos_img[k])))
            
            print(f"Evento salvo: {nome_pasta}") 
            state['salvos'] += 1
            state['evt_idx'] += 1
            atualizar_evento()

        def on_key(event):
            k = event.key
            
            if k == 'right':
                if state['frame_idx'] < len(arquivos_img) - 1:
                    state['frame_idx'] += 1
                    renderizar_frame()
            elif k == 'left':
                if state['frame_idx'] > 0:
                    state['frame_idx'] -= 1
                    renderizar_frame()
            elif k == 'up':
                state['frame_idx'] = state['inicio']
                renderizar_frame()
            elif k == 'down':
                state['frame_idx'] = state['fim']
                renderizar_frame()
            elif k == 'p':
                state['inicio'] = state['frame_idx']
                print(f"Novo inicio: {state['inicio']}")
            elif k == 'u':
                state['fim'] = state['frame_idx']
                print(f"Novo fim: {state['fim']}")
            
            elif k == 'c': salvar_selecao('CG')
            elif k == 'i': salvar_selecao('IC')
            elif k == 'l': salvar_selecao('LCC')
            elif k == 'b': salvar_selecao('Brilho')
            
            elif k in ['n', 'd']:
                state['evt_idx'] += 1
                atualizar_evento()
            elif k == 'a':
                state['evt_idx'] -= 1
                atualizar_evento()
            elif k == 'q':
                state['running'] = False
                plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        
        # Iniciar ciclo
        atualizar_evento()
        plt.show(block=True)
        self.log(f"Edição encerrada. Total salvos: {state['salvos']}")

if __name__ == "__main__":
    # Necessário para Windows + Multiprocessing
    multiprocessing.freeze_support()
    
    root = tk.Tk()
    app = AppAnaliseRaios(root)
    root.mainloop()

