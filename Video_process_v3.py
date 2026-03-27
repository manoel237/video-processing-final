import cv2
import numpy as np
import glob
import os
import shutil
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import time
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

# --- WORKER DE PROCESSAMENTO (PARALELO) ---
# Deve ficar fora das classes para o multiprocessing funcionar no Windows
def worker_analisar_frame(args):
    img_path, vetor_ref_v, vetor_ref_h, n_v, n_h = args
    
    img_cinza = cv2.imread(img_path, 0) 
    if img_cinza is None: 
        return (None, None, None, None)
    
    cortes_v = np.array_split(img_cinza, n_v, axis=1)
    lums_rel_v = np.array([np.mean(c) for c in cortes_v]) - vetor_ref_v
    
    cortes_h = np.array_split(img_cinza, n_h, axis=0)
    lums_rel_h = np.array([np.mean(c) for c in cortes_h]) - vetor_ref_h
    
    std_dev = np.std(lums_rel_v)
    max_val = np.max(lums_rel_v)
    
    return (std_dev, max_val, lums_rel_v, lums_rel_h)

# ==========================================
# CLASSE PRINCIPAL (CONTAINER)
# ==========================================
class DetectorRaiosApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("⚡ DETECTOR DE RAIOS - COM EXPLICITADOR DE REGIÃO ⚡")
        self.geometry("1000x750")
        self.configure(bg="#101010")

        style = ttk.Style()
        style.theme_use('clam')
        style.configure("TFrame", background="#101010")
        style.configure("TLabel", background="#101010", foreground="white", font=("Arial", 10))
        style.configure("Header.TLabel", font=("Impact", 16), foreground="#00ff00")
        
        self.container = ttk.Frame(self)
        self.container.pack(fill="both", expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)

        self.configuracoes_atuais = {}
        self.frames = {}

        for Tela in (TelaConfiguracoes, TelaAnalise):
            nome_tela = Tela.__name__
            frame = Tela(parent=self.container, controller=self)
            self.frames[nome_tela] = frame
            frame.grid(row=0, column=0, sticky="nsew")

        self.mostrar_tela("TelaConfiguracoes")

    def mostrar_tela(self, nome_tela):
        frame = self.frames[nome_tela]
        frame.tkraise()
        # Se a tela tiver um método específico para quando for mostrada, executa ele
        if hasattr(frame, 'ao_mostrar_tela'):
            frame.ao_mostrar_tela()

# ==========================================
# TELA 1: CONFIGURAÇÕES INICIAIS
# ==========================================
class TelaConfiguracoes(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller

        lbl_titulo = ttk.Label(self, text="CONFIGURAÇÕES DE PARÂMETROS", style="Header.TLabel")
        lbl_titulo.pack(pady=20)

        frame_form = tk.Frame(self, bg="#151515", padx=20, pady=20)
        frame_form.pack(pady=10)

        self.var_pasta_in = tk.StringVar(value=r"D:\videos_novos\convertido")
        self.var_pasta_out = tk.StringVar(value=r"D:\videos_novos\classificados")
        self.var_sufixo = tk.StringVar(value="_classificado")
        self.var_processos = tk.StringVar(value="None")
        
        self.var_tempo = tk.StringVar(value="1.266247")
        self.var_bg_frames = tk.StringVar(value="75")
        self.var_cortes_v = tk.StringVar(value="336")
        self.var_cortes_h = tk.StringVar(value="252")
        self.var_percentil = tk.StringVar(value="75")
        self.var_gap = tk.StringVar(value="4")
        self.var_folga = tk.StringVar(value="1")

        def criar_linha_texto(parent, label_text, tk_var, row):
            ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="e", pady=5, padx=5)
            tk.Entry(parent, textvariable=tk_var, width=20, bg="#333", fg="white", font=("Consolas", 10)).grid(row=row, column=1, sticky="w", pady=5)

        def criar_linha_pasta(parent, label_text, tk_var, row):
            ttk.Label(parent, text=label_text).grid(row=row, column=0, sticky="e", pady=5, padx=5)
            tk.Entry(parent, textvariable=tk_var, width=50, bg="#333", fg="white", font=("Consolas", 10)).grid(row=row, column=1, sticky="w", pady=5)
            tk.Button(parent, text="Procurar...", command=lambda: selecionar_pasta(tk_var), bg="#444", fg="white", relief="flat").grid(row=row, column=2, padx=5)

        def selecionar_pasta(tk_var):
            pasta = filedialog.askdirectory()
            if pasta:
                tk_var.set(pasta)

        ttk.Label(frame_form, text="DIRETÓRIOS", font=("Arial", 11, "bold"), foreground="#00aaaa").grid(row=0, column=0, columnspan=3, pady=(0,10))
        criar_linha_pasta(frame_form, "Pasta Vídeos Novos:", self.var_pasta_in, 1)
        criar_linha_pasta(frame_form, "Pasta Raiz Saídas:", self.var_pasta_out, 2)
        criar_linha_texto(frame_form, "Sufixo da Saída:", self.var_sufixo, 3)

        ttk.Label(frame_form, text="\nPARÂMETROS FÍSICOS & TUNING", font=("Arial", 11, "bold"), foreground="#00aaaa").grid(row=4, column=0, columnspan=3, pady=(10,10))
        criar_linha_texto(frame_form, "Núcleos Paralelos (None=Todos):", self.var_processos, 5)
        criar_linha_texto(frame_form, "Tempo Gravação (s):", self.var_tempo, 6)
        criar_linha_texto(frame_form, "Frames Background:", self.var_bg_frames, 7)
        criar_linha_texto(frame_form, "Cortes Verticais:", self.var_cortes_v, 8)
        criar_linha_texto(frame_form, "Cortes Horizontais:", self.var_cortes_h, 9)
        criar_linha_texto(frame_form, "Percentil:", self.var_percentil, 10)
        criar_linha_texto(frame_form, "Max Gap Frames:", self.var_gap, 11)
        criar_linha_texto(frame_form, "Folga Frames:", self.var_folga, 12)

        btn_avancar = tk.Button(self, text="SALVAR & AVANÇAR PARA ANÁLISE ➔", command=self.salvar_e_avancar,
                                bg="#006600", fg="white", font=("Arial", 12, "bold"), relief="flat", padx=20, pady=10)
        btn_avancar.pack(pady=30)

    def salvar_e_avancar(self):
        try:
            self.controller.configuracoes_atuais = {
                "PASTA_VIDEOS_NOVOS": self.var_pasta_in.get(),
                "PASTA_RAIZ_SAIDAS": self.var_pasta_out.get(),
                "SUFIXO_PASTA_SAIDA": self.var_sufixo.get(),
                "NUM_PROCESSOS_PARALELOS": None if self.var_processos.get().lower() == "none" else int(self.var_processos.get()),
                "TEMPO_DE_GRAVACAO_SEGUNDOS": float(self.var_tempo.get()),
                "NUM_FRAMES_BACKGROUND": int(self.var_bg_frames.get()),
                "NUMERO_DE_CORTES_VERTICAL": int(self.var_cortes_v.get()),
                "NUMERO_DE_CORTES_HORIZONTAL": int(self.var_cortes_h.get()),
                "PERCENTIL": float(self.var_percentil.get()),
                "MAX_GAP_ENTRE_FRAMES": int(self.var_gap.get()),
                "FOLGA_FRAMES": int(self.var_folga.get())
            }
            self.controller.mostrar_tela("TelaAnalise")
        except ValueError:
            messagebox.showerror("Erro de Digitação", "Verifique se você digitou apenas números nos campos de parâmetros físicos.")

# ==========================================
# TELA 2: PAINEL DE ANÁLISE
# ==========================================
class TelaAnalise(ttk.Frame):
    def __init__(self, parent, controller):
        super().__init__(parent)
        self.controller = controller
        
        plt.style.use('dark_background')

        # --- BARRA SUPERIOR ---
        frame_topo = tk.Frame(self, bg="#222", pady=10, padx=10)
        frame_topo.pack(fill=tk.X)

        btn_voltar = tk.Button(frame_topo, text="⬅ VOLTAR ÀS CONFIGURAÇÕES", command=lambda: controller.mostrar_tela("TelaConfiguracoes"),
                               bg="#555", fg="white", font=("Arial", 10, "bold"), relief="flat", padx=10)
        btn_voltar.pack(side=tk.LEFT)

        lbl_titulo = tk.Label(frame_topo, text="PAINEL DE EXECUÇÃO", bg="#222", fg="#00ff00", font=("Impact", 14))
        lbl_titulo.pack(side=tk.LEFT, padx=20)

        # --- CONTEÚDO PRINCIPAL (SPLIT) ---
        self.paned = tk.PanedWindow(self, orient=tk.HORIZONTAL, bg="#101010", sashwidth=4)
        self.paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # PAINEL ESQUERDO (LISTA)
        frame_left = tk.Frame(self.paned, bg="#101010")
        self.paned.add(frame_left, minsize=400)
        
        lbl_list = tk.Label(frame_left, text="LISTA DE VÍDEOS", bg="#101010", fg="#00ff00", font=("Impact", 12))
        lbl_list.pack(pady=5)
        
        style = ttk.Style()
        style.configure("Treeview", background="#202020", foreground="#00ff00", fieldbackground="#202020", font=('Consolas', 10))
        style.configure("Treeview.Heading", background="#333", foreground="white", font=('Arial', 10, 'bold'))
        style.map('Treeview', background=[('selected', '#004400')])

        self.tree = ttk.Treeview(frame_left, columns=("num", "nome", "status"), show='headings', selectmode='browse')
        self.tree.heading("num", text="#")
        self.tree.column("num", width=40, anchor='center')
        self.tree.heading("nome", text="Arquivo")
        self.tree.column("nome", width=300, anchor='w')
        self.tree.heading("status", text="Estado")
        self.tree.column("status", width=80, anchor='center')
        
        self.tree.tag_configure('concluido', background='#003300', foreground='#00ff00')
        self.tree.tag_configure('pendente', foreground='white')
        self.tree.pack(fill=tk.BOTH, expand=True, padx=5)
        
        scroll_tree = ttk.Scrollbar(frame_left, orient="vertical", command=self.tree.yview)
        scroll_tree.place(relx=0.96, rely=0.05, relheight=0.95)
        self.tree.configure(yscroll=scroll_tree.set)

        # PAINEL DIREITO (LOGS E BOTÕES)
        frame_right = tk.Frame(self.paned, bg="#151515")
        self.paned.add(frame_right, minsize=400)
        
        lbl_log = tk.Label(frame_right, text="LOG DE SISTEMA", bg="#151515", fg="#00ff00", font=("Impact", 12))
        lbl_log.pack(pady=5)
        
        self.log_text = scrolledtext.ScrolledText(frame_right, bg="black", fg="#00ff00", font=("Consolas", 9), state='disabled')
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5)
        
        frame_btns = tk.Frame(frame_right, bg="#151515")
        frame_btns.pack(pady=10, fill=tk.X)
        
        btn_refresh = tk.Button(frame_btns, text="ATUALIZAR LISTA", command=self.carregar_lista, 
                                bg="#333", fg="white", font=("Arial", 10, "bold"), relief="flat", padx=10, pady=5)
        btn_refresh.pack(side=tk.LEFT, padx=10)
        
        btn_start = tk.Button(frame_btns, text="▶ INICIAR ANÁLISE", command=self.iniciar_processamento, 
                              bg="#006600", fg="white", font=("Arial", 11, "bold"), relief="flat", padx=20, pady=5)
        btn_start.pack(side=tk.RIGHT, padx=10)

        self.mapa_caminhos = {}

    def log(self, msg):
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, f">> {msg}\n")
        self.log_text.see(tk.END)
        self.log_text.config(state='disabled')
        self.controller.update()

    def ao_mostrar_tela(self):
        """Disparado automaticamente sempre que essa tela vem para a frente"""
        self.log_text.config(state='normal')
        self.log_text.delete(1.0, tk.END)
        self.log_text.config(state='disabled')
        self.log("Lendo configurações...")
        cpu_count = multiprocessing.cpu_count()
        self.log(f"Núcleos de CPU detectados: {cpu_count}")
        self.carregar_lista()

    def carregar_lista(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.mapa_caminhos = {}
        
        pasta_alvo = self.controller.configuracoes_atuais.get("PASTA_VIDEOS_NOVOS", "")
        
        if not os.path.exists(pasta_alvo):
            self.log(f"ERRO: Pasta não encontrada: {pasta_alvo}")
            return

        pastas = sorted([d for d in os.listdir(pasta_alvo) if os.path.isdir(os.path.join(pasta_alvo, d))])
        
        for i, nome_pasta in enumerate(pastas):
            iid = self.tree.insert("", "end", values=(f"{i+1:02d}", nome_pasta, "Pendente"), tags=('pendente',))
            self.mapa_caminhos[iid] = os.path.join(pasta_alvo, nome_pasta)
            
        self.log(f"Lista atualizada: {len(pastas)} vídeos encontrados.")

    def iniciar_processamento(self):
        selected = self.tree.selection()
        if not selected:
            messagebox.showwarning("Aviso", "Selecione um vídeo na lista!")
            return
        
        item_id = selected[0]
        caminho_pasta = self.mapa_caminhos[item_id]
        
        self.tree.set(item_id, "status", "Rodando...")
        self.controller.config(cursor="watch")
        
        try:
            sucesso = self.executar_analise_video(caminho_pasta)
            if sucesso:
                self.tree.set(item_id, "status", "Concluído")
                self.tree.item(item_id, tags=('concluido',))
                self.log(f"Sucesso. Vídeo processado.")
            else:
                self.tree.set(item_id, "status", "Vazio/Erro")
        except Exception as e:
            self.log(f"ERRO CRÍTICO: {e}")
            messagebox.showerror("Erro", str(e))
        finally:
            self.controller.config(cursor="")

    def executar_analise_video(self, caminho_pasta_entrada):
        cfg = self.controller.configuracoes_atuais
        
        nome_base = os.path.basename(os.path.normpath(caminho_pasta_entrada))
        self.log(f"--- INICIANDO: {nome_base} ---")
        
        arquivos_img = sorted(glob.glob(os.path.join(caminho_pasta_entrada, '*.jpg')))
        total_frames = len(arquivos_img)
        
        if total_frames < cfg["NUM_FRAMES_BACKGROUND"]:
            self.log("Vídeo muito curto.")
            return False

        self.log("Calibrando background...")
        vetor_v, vetor_h = self.calcular_vetores_referencia(arquivos_img, cfg)
        
        self.log(f"Iniciando Scan em Paralelo ({total_frames} frames)...")
        t_inicio = time.time()
        
        args_list = [
            (f, vetor_v, vetor_h, cfg["NUMERO_DE_CORTES_VERTICAL"], cfg["NUMERO_DE_CORTES_HORIZONTAL"]) 
            for f in arquivos_img
        ]
        
        dados_processados = []
        with ProcessPoolExecutor(max_workers=cfg["NUM_PROCESSOS_PARALELOS"]) as executor:
            resultados = executor.map(worker_analisar_frame, args_list)
            for i, res in enumerate(resultados):
                dados_processados.append(res)
                if i % 100 == 0:
                    self.controller.update()
        
        tempo_total = time.time() - t_inicio
        self.log(f"Scan concluído: {tempo_total:.2f}s ({(total_frames / tempo_total):.1f} FPS).")

        all_std, all_max, final_dados = [], [], []
        
        for dp, mx, vv, vh in dados_processados:
            if dp is not None:
                all_std.append(dp)
                all_max.append(mx)
                final_dados.append((vv, vh))
            else:
                all_std.append(0)
                all_max.append(0)
                final_dados.append((np.zeros(cfg["NUMERO_DE_CORTES_VERTICAL"]), np.zeros(cfg["NUMERO_DE_CORTES_HORIZONTAL"])))

        limiar_std = np.percentile(all_std, cfg["PERCENTIL"])
        limiar_max = np.percentile(all_max, cfg["PERCENTIL"])
        
        indices_chave = [i for i in range(len(all_std)) if all_std[i] > limiar_std and all_max[i] > limiar_max]
        eventos = self.agrupar_em_eventos(indices_chave, total_frames, cfg)
        
        self.log(f"Eventos encontrados: {len(eventos)}")
        
        if not eventos: return False

        self.abrir_editor_visual(eventos, arquivos_img, all_max, final_dados, nome_base, cfg)
        return True

    def calcular_vetores_referencia(self, arquivos_img, cfg):
        frames = arquivos_img[:cfg["NUM_FRAMES_BACKGROUND"]]
        soma_v = np.zeros(cfg["NUMERO_DE_CORTES_VERTICAL"])
        soma_h = np.zeros(cfg["NUMERO_DE_CORTES_HORIZONTAL"])
        count = 0
        for p in frames:
            img = cv2.imread(p, 0)
            if img is None: continue
            parts_v = np.array_split(img, cfg["NUMERO_DE_CORTES_VERTICAL"], axis=1)
            soma_v += np.array([np.mean(c) for c in parts_v])
            parts_h = np.array_split(img, cfg["NUMERO_DE_CORTES_HORIZONTAL"], axis=0)
            soma_h += np.array([np.mean(c) for c in parts_h])
            count += 1
        return (soma_v / count), (soma_h / count)

    def agrupar_em_eventos(self, indices, total, cfg):
        if not indices: return []
        evs = []
        atual = [indices[0]]
        for i in range(1, len(indices)):
            if indices[i] - atual[-1] <= cfg["MAX_GAP_ENTRE_FRAMES"] + 1:
                atual.append(indices[i])
            else:
                evs.append((max(0, min(atual) - cfg["FOLGA_FRAMES"]), min(total - 1, max(atual) + cfg["FOLGA_FRAMES"])))
                atual = [indices[i]]
        evs.append((max(0, min(atual) - cfg["FOLGA_FRAMES"]), min(total - 1, max(atual) + cfg["FOLGA_FRAMES"])))
        return evs

    def abrir_editor_visual(self, eventos, arquivos_img, all_max, dados_vetores, nome_video, cfg):
        self.log("Abrindo Editor Visual...")
        
        pasta_saida_final = os.path.join(cfg["PASTA_RAIZ_SAIDAS"], f"{nome_video}{cfg['SUFIXO_PASTA_SAIDA']}")
        os.makedirs(pasta_saida_final, exist_ok=True)
        
        state = {'evt_idx': 0, 'frame_idx': 0, 'inicio': 0, 'fim': 0, 'salvos': 0, 'running': True}
        fig = plt.figure(figsize=(16, 9))
        try:
            plt.get_current_fig_manager().window.state('zoomed') 
        except Exception: pass 

        gs = gridspec.GridSpec(4, 5, figure=fig)
        ax_img = fig.add_subplot(gs[0:3, 0:4])
        ax_v = fig.add_subplot(gs[3, 0:4])
        ax_h = fig.add_subplot(gs[0:3, 4])
        plt.subplots_adjust(left=0.05, right=0.98, top=0.92, bottom=0.08, wspace=0.1, hspace=0.1)

        im_obj = ax_img.imshow(np.zeros((100, 100, 3), dtype=np.uint8))
        ax_img.axis('off')
        target_box = patches.Rectangle((0, 0), 10, 10, linewidth=3, edgecolor='#FF0000', facecolor='none', alpha=0.9)
        ax_img.add_patch(target_box)
        
        bar_v = ax_v.bar(np.arange(cfg["NUMERO_DE_CORTES_VERTICAL"]), np.zeros(cfg["NUMERO_DE_CORTES_VERTICAL"]))
        ax_v.grid(axis='y', linestyle=':', alpha=0.5)
        bar_h = ax_h.barh(np.arange(cfg["NUMERO_DE_CORTES_HORIZONTAL"]), np.zeros(cfg["NUMERO_DE_CORTES_HORIZONTAL"]))
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
            state['frame_idx'] = inicio + np.argmax(sub_max)
            state['inicio'] = inicio
            state['fim'] = fim
            renderizar_frame()

        def renderizar_frame():
            idx = state['frame_idx']
            v_v, v_h = dados_vetores[idx]
            
            bgr = cv2.imread(arquivos_img[idx])
            img_h, img_w = 0, 0
            if bgr is not None:
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                img_h, img_w, _ = rgb.shape
                im_obj.set_data(rgb)
                im_obj.set_extent((0, img_w, img_h, 0))

            if img_w > 0 and img_h > 0:
                vv_abs, vh_abs = np.abs(v_v), np.abs(v_h)
                threshold_v = max(5.0, np.max(vv_abs) * 0.3)
                threshold_h = max(5.0, np.max(vh_abs) * 0.3)
                
                indices_ativos_v = np.where(vv_abs > threshold_v)[0]
                indices_ativos_h = np.where(vh_abs > threshold_h)[0]
                
                largura_bloco = img_w / cfg["NUMERO_DE_CORTES_VERTICAL"]
                altura_bloco = img_h / cfg["NUMERO_DE_CORTES_HORIZONTAL"]
                
                box_x, box_y, box_w, box_h = 0, 0, img_w, img_h
                tem_sinal_v, tem_sinal_h = len(indices_ativos_v) > 0, len(indices_ativos_h) > 0
                
                if tem_sinal_v:
                    box_x = indices_ativos_v[0] * largura_bloco
                    box_w = (indices_ativos_v[-1] - indices_ativos_v[0] + 1) * largura_bloco
                if tem_sinal_h:
                    box_y = indices_ativos_h[0] * altura_bloco
                    box_h = (indices_ativos_h[-1] - indices_ativos_h[0] + 1) * altura_bloco
                
                if tem_sinal_v or tem_sinal_h:
                    margin = 20
                    draw_x = max(0, box_x - margin/2)
                    draw_y = max(0, box_y - margin/2)
                    target_box.set_xy((draw_x, draw_y))
                    target_box.set_width(min(img_w - draw_x, box_w + margin))
                    target_box.set_height(min(img_h - draw_y, box_h + margin))
                    target_box.set_alpha(1.0)
                else:
                    target_box.set_alpha(0.0)

            limite_v = max(np.max(np.abs(v_v)) * 1.1, 1.0) 
            ax_v.set_ylim(-limite_v, limite_v)
            for rect, val in zip(bar_v, v_v):
                rect.set_height(val)
                rect.set_color('#00ff00' if val >= 0 else '#ff0000')

            limite_h = max(np.max(np.abs(v_h)) * 1.1, 1.0)
            ax_h.set_xlim(-limite_h, limite_h)
            for rect, val in zip(bar_h, v_h):
                rect.set_width(val)
                rect.set_color('#00ff00' if val >= 0 else '#ff0000')

            texto_titulo.set_text(f"EVENTO {state['evt_idx']+1}/{len(eventos)} | Frame: {idx} | Range: {state['inicio']}-{state['fim']}\nCOMANDOS: Setas(Nav) | P/U(Range) | C/I/L/B(Salvar) | N(Pular)")
            fig.canvas.draw_idle()

        def salvar_selecao(tipo):
            dur_ms = (state['fim'] - state['inicio'] + 1) * ((cfg["TEMPO_DE_GRAVACAO_SEGUNDOS"] * 1000) / len(arquivos_img))
            ref_idx = state['frame_idx']
            ts = os.path.splitext(os.path.basename(arquivos_img[ref_idx]))[0].split('_', 1)[1]
            
            nome_pasta = f"{ref_idx:04d} {tipo} {ts} Dur {dur_ms:.1f}ms"
            path_evt = os.path.join(pasta_saida_final, nome_pasta)
            os.makedirs(path_evt, exist_ok=True)
            
            fig.savefig(os.path.join(path_evt, f"Info_Frame_{ref_idx}.png"), bbox_inches='tight')
            for k in range(state['inicio'], state['fim'] + 1):
                shutil.copy(arquivos_img[k], os.path.join(path_evt, os.path.basename(arquivos_img[k])))
            
            print(f"Salvo: {nome_pasta}") 
            state['salvos'] += 1
            state['evt_idx'] += 1
            atualizar_evento()

        def on_key(event):
            k = event.key
            if k == 'right' and state['frame_idx'] < len(arquivos_img) - 1: state['frame_idx'] += 1; renderizar_frame()
            elif k == 'left' and state['frame_idx'] > 0: state['frame_idx'] -= 1; renderizar_frame()
            elif k == 'up': state['frame_idx'] = state['inicio']; renderizar_frame()
            elif k == 'down': state['frame_idx'] = state['fim']; renderizar_frame()
            elif k == 'p': state['inicio'] = state['frame_idx']; print(f"Inicio: {state['inicio']}")
            elif k == 'u': state['fim'] = state['frame_idx']; print(f"Fim: {state['fim']}")
            elif k == 'c': salvar_selecao('CG')
            elif k == 'i': salvar_selecao('IC')
            elif k == 'l': salvar_selecao('LCC')
            elif k == 'b': salvar_selecao('Brilho')
            elif k in ['n', 'd']: state['evt_idx'] += 1; atualizar_evento()
            elif k == 'a': state['evt_idx'] -= 1; atualizar_evento()
            elif k == 'q': state['running'] = False; plt.close(fig)

        fig.canvas.mpl_connect('key_press_event', on_key)
        atualizar_evento()
        plt.show(block=True)
        self.log(f"Edição encerrada. Total salvos: {state['salvos']}")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    app = DetectorRaiosApp()
    app.mainloop()
