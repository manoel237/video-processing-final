import os

def listar_pastas(pasta_entrada, pasta_saida, nome_arquivo_saida="nomes_pastas.txt"):
    # 1. Verifica se a pasta de entrada existe
    if not os.path.exists(pasta_entrada):
        print(f"Erro: A pasta de entrada '{pasta_entrada}' não foi encontrada.")
        return

    # 2. Cria a pasta de saída se ela não existir
    if not os.path.exists(pasta_saida):
        os.makedirs(pasta_saida)
        print(f"Pasta de saída '{pasta_saida}' criada.")

    try:
        # 3. Lista todos os itens e filtra apenas o que for diretório (pasta)
        # O uso de os.path.join é necessário para verificar o caminho completo
        nomes_das_pastas = [
            item for item in os.listdir(pasta_entrada)
            if os.path.isdir(os.path.join(pasta_entrada, item))
        ]
        
        # Caminho completo do arquivo de texto final
        caminho_arquivo_final = os.path.join(pasta_saida, nome_arquivo_saida)

        # 4. Escreve os nomes no arquivo .txt
        with open(caminho_arquivo_final, 'w', encoding='utf-8') as f:
            for nome in nomes_das_pastas:
                f.write(nome + '\n')

        print(f"Sucesso! Foram listadas {len(nomes_das_pastas)} pastas.")
        print(f"Arquivo salvo em: {caminho_arquivo_final}")

    except Exception as e:
        print(f"Ocorreu um erro: {e}")

# --- CONFIGURAÇÃO ---
# Altere os caminhos abaixo conforme o seu computador.
# Dica: No Windows, use barras duplas (\\) ou uma barra invertida (/)
caminho_entrada = r"C:\Exemplo\PastaOrigem" 
caminho_saida = r"C:\Exemplo\PastaDestino"

# Executa a função
listar_pastas(caminho_entrada, caminho_saida)