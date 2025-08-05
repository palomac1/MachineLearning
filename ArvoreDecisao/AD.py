import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------------------------
# SECAO 1: PRE-PROCESSAMENTO DOS DADOS PARA OBTENCAO DO DATASET 
# -------------------------------------------------------------------

def criar_dataset_processado():
    """
    Carrega o dataset Pima Indians, aplica o pre-processamento,
    salva o resultado e retorna o caminho do arquivo salvo.
    """
    nome_arquivo_entrada = "diabetes.csv"
    nome_arquivo_saida = "diabetes_processado.csv"

    if os.path.exists(nome_arquivo_saida):
        print(f"O arquivo '{nome_arquivo_saida}' ja existe. Usando o arquivo existente.")
        return nome_arquivo_saida
    else:
        try:
            dataframe = pd.read_csv(nome_arquivo_entrada)
            print(f"Dataset '{nome_arquivo_entrada}' carregado com sucesso.")
        except FileNotFoundError:
            print(f"Erro: O arquivo de entrada '{nome_arquivo_entrada}' nao foi encontrado.")
            print("Por favor, certifique-se de que ele esta na mesma pasta do script.")
            return None

        print("Iniciando a criacao do dataset pre-processado...")

        # Verifica se o dataset possui colunas esperadas
        cols_com_zeros_invalidos = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        for coluna in cols_com_zeros_invalidos:
            dataframe[coluna] = pd.to_numeric(dataframe[coluna], errors='coerce') # Converte para numérico, tratando erros como NaN (Not a Number)
            dataframe[coluna] = dataframe[coluna].replace(0, np.nan) # Substitui zeros por NaN para evitar distorções
            mediana = dataframe[coluna].median() # Calcula a mediana ignorando NaNs já existentes
            dataframe[coluna] = dataframe[coluna].fillna(mediana) # Preenche NaNs com a mediana 

        faixas = [0, 29, 59, 150] # Definindo os limites das faixas etárias
        rotulos = ['Jovem', 'Adulto', 'Idoso'] # Definindo os rótulos para as faixas etárias
        dataframe['Faixa_Etaria'] = pd.cut(dataframe['Age'], bins=faixas, labels=rotulos, right=True) # Cria a coluna de faixa etária 

        print(f"\nDataset processado possui {len(dataframe)} instancias.")
        dataframe.to_csv(nome_arquivo_saida, index=False)
        print(f"Dataset pre-processado foi salvo com sucesso como '{nome_arquivo_saida}'")
        
        return nome_arquivo_saida

# -------------------------------------------------------------------
# SECAO 2: IMPLEMENTACAO DA ARVORE DE DECISAO PARA CLASSIFICACAO
# -------------------------------------------------------------------

def calcular_entropia(y):  #Função que calcula a entropia de um conjunto de rótulos
    contagem_classes = np.bincount(y) # Conta a ocorrência de cada classe para calcular a entropia ()
    probabilidades = contagem_classes / len(y) # Calcula a probabilidade de cada classe
    probabilidades = probabilidades[probabilidades > 0] # Remove probabilidades zero para evitar log(0)
    return -np.sum(probabilidades * np.log2(probabilidades)) 

def calcular_ganho_de_informacao(y_pai, y_esquerda, y_direita): #Função que calcula o ganho de informação entre o nó pai e os nós filhos
    peso_esquerda = len(y_esquerda) / len(y_pai) # Calcula o peso da divisão esquerda para o ganho de informação para ponderar a entropia
    peso_direita = len(y_direita) / len(y_pai) # Calcula o peso da divisão direita
    entropia_pai = calcular_entropia(y_pai) # Entropia do nó pai para calcular o ganho de informação
    entropia_ponderada_filhos = (peso_esquerda * calcular_entropia(y_esquerda) +
                                 peso_direita * calcular_entropia(y_direita)) # Entropia ponderada dos nós filhos
    return entropia_pai - entropia_ponderada_filhos

def calcular_gini(y):
    contagem_classes = np.bincount(y) # Conta a ocorrência de cada classe para calcular o índice de Gini
    probabilidades = contagem_classes / len(y) # Calcula a probabilidade de cada classe
    return 1 - np.sum(probabilidades**2) # Calcula o índice de Gini como 1 menos a soma dos quadrados das probabilidades para evitar distorções

def calcular_ganho_gini(y_pai, y_esquerda, y_direita): #Função que calcula o ganho de Gini entre o nó pai e os nós filhos
    peso_esquerda = len(y_esquerda) / len(y_pai) # Calcula o peso da divisão esquerda para o ganho de Gini
    peso_direita = len(y_direita) / len(y_pai) # Calcula o peso da divisão direita para o ganho de Gini 
    gini_pai = calcular_gini(y_pai) # Índice de Gini do nó pai para calcular o ganho de Gini
    gini_ponderado_filhos = (peso_esquerda * calcular_gini(y_esquerda) +
                             peso_direita * calcular_gini(y_direita))
    return gini_pai - gini_ponderado_filhos

def calcular_razao_de_ganho(y_pai, y_esquerda, y_direita):
    ganho_info = calcular_ganho_de_informacao(y_pai, y_esquerda, y_direita) # Calcula o ganho de informação para a razão de ganho
    if ganho_info == 0: return 0
    proporcao_esquerda = len(y_esquerda) / len(y_pai) # Calcula a proporção da divisão esquerda
    proporcao_direita = len(y_direita) / len(y_pai) # Calcula a proporção da divisão direita
    info_divisao = - (proporcao_esquerda * np.log2(proporcao_esquerda + 1e-9) +
                      proporcao_direita * np.log2(proporcao_direita + 1e-9)) # Calcula a informação da divisão para a razão de ganho
    if info_divisao == 0: return 0
    return ganho_info / info_divisao

class No:
    def __init__(self, atributo=None, limiar=None, esquerda=None, direita=None, *, valor=None): #
        # Atributo que define a divisão do nó, limiar de divisão, subárvore esquerda, subárvore direita e valor do nó folha
        self.atributo = atributo
        self.limiar = limiar
        self.esquerda = esquerda
        self.direita = direita
        self.valor = valor
    def e_no_folha(self):
        return self.valor is not None

class ArvoreDeDecisao:
    # Classe que implementa a árvore de decisão para classificação
    # Possui parâmetros para controle de divisão, profundidade máxima e critério de divisão
    def __init__(self, min_amostras_divisao=2, profundidade_maxima=100, criterio='information_gain'):
        self.min_amostras_divisao = min_amostras_divisao
        self.profundidade_maxima = profundidade_maxima
        self.criterio = criterio
        self.raiz = None
        
    # Método para treinar a árvore de decisão com os dados de entrada e rótulos
    # Constrói a árvore recursivamente, dividindo os dados até atingir as condições de parada
    def fit(self, X, y): 
        valores_X = X.values if isinstance(X, pd.DataFrame) else X
        valores_y = y.values if isinstance(y, pd.Series) else y
        self.raiz = self._construir_arvore(valores_X, valores_y)

    # Método recursivo que constrói a árvore de decisão
    # Verifica condições de parada como profundidade máxima, número de rótulos únicos e número de amostras
    # Se as condições forem atendidas, cria um nó folha com o rótulo mais comum
    # Caso contrário, encontra a melhor divisão dos dados com base no critério especificado
    def _construir_arvore(self, X, y, profundidade=0):
        n_amostras, n_atributos = X.shape
        n_rotulos = len(np.unique(y))
        
        if (profundidade >= self.profundidade_maxima or n_rotulos == 1 or n_amostras < self.min_amostras_divisao):
            valor_folha = self._rotulo_mais_comum(y)
            return No(valor=valor_folha)
            
        indices_atributos = np.random.choice(n_atributos, n_atributos, replace=False)
        melhor_divisao = self._encontrar_melhor_divisao(X, y, indices_atributos)
        
        if not melhor_divisao or melhor_divisao['ganho'] <= 0:
            valor_folha = self._rotulo_mais_comum(y)
            return No(valor=valor_folha)
            
        indices_esquerda, indices_direita = melhor_divisao['indices']
        arvore_esquerda = self._construir_arvore(X[indices_esquerda, :], y[indices_esquerda], profundidade + 1)
        arvore_direita = self._construir_arvore(X[indices_direita, :], y[indices_direita], profundidade + 1)
        
        return No(melhor_divisao['atributo'], melhor_divisao['limiar'], arvore_esquerda, arvore_direita)

    # Método que encontra a melhor divisão dos dados com base no critério especificado
    # Itera sobre os atributos e seus valores únicos, calculando o ganho de informação,
    # razão de ganho ou ganho de Gini para cada possível divisão
    def _encontrar_melhor_divisao(self, X, y, indices_atributos):
        melhor_ganho = -1
        melhor_divisao = None
        for indice_atributo in indices_atributos:
            limiares = np.unique(X[:, indice_atributo])
            for limiar in limiares:
                indices_esquerda = np.where(X[:, indice_atributo] <= limiar)[0]
                indices_direita = np.where(X[:, indice_atributo] > limiar)[0]
                
                if len(indices_esquerda) == 0 or len(indices_direita) == 0: continue
                
                if self.criterio == 'information_gain':
                    ganho = calcular_ganho_de_informacao(y, y[indices_esquerda], y[indices_direita])
                elif self.criterio == 'gain_ratio':
                    ganho = calcular_razao_de_ganho(y, y[indices_esquerda], y[indices_direita])
                else: # 'gini'
                    ganho = calcular_ganho_gini(y, y[indices_esquerda], y[indices_direita])
                    
                if ganho > melhor_ganho:
                    melhor_ganho = ganho
                    melhor_divisao = {'atributo': indice_atributo, 'limiar': limiar,
                                      'indices': (indices_esquerda, indices_direita), 'ganho': ganho}
        return melhor_divisao

    # Método que retorna o rótulo mais comum entre os rótulos fornecidos
    # Utiliza a classe Counter para contar as ocorrências de cada rótulo e retorna
    def _rotulo_mais_comum(self, y):
        contador = Counter(y)
        return contador.most_common(1)[0][0]

    # Método que percorre a árvore de decisão para fazer previsões
    # Utiliza a função recursiva _percorrer_arvore para navegar pela árvore e retornar o valor do nó folha correspondente a cada entrada
    def predict(self, X):
        valores_X = X.values if isinstance(X, pd.DataFrame) else X
        return np.array([self._percorrer_arvore(x, self.raiz) for x in valores_X])

    # Método recursivo que percorre a árvore de decisão para fazer previsões
    # Verifica se o nó atual é um nó folha e retorna seu valor, caso contrário verifica o atributo e o limiar para decidir se deve seguir para a subárvore esquerda ou direita
    def _percorrer_arvore(self, x, no):
        if no.e_no_folha(): return no.valor
        if x[no.atributo] <= no.limiar:
            return self._percorrer_arvore(x, no.esquerda)
        return self._percorrer_arvore(x, no.direita)

# -------------------------------------------------------------------
# SECAO 3: EXECUCAO PRINCIPAL E TESTES PARA AVALIACAO DO MODELO
# -------------------------------------------------------------------

def principal():
    
    # Cria o dataset processado e carrega os dados 
    caminho_arquivo_processado = criar_dataset_processado() 
    if caminho_arquivo_processado is None:
        return 
    
    try:
        dataframe_processado = pd.read_csv(caminho_arquivo_processado) 
        print(f"\nDataset '{caminho_arquivo_processado}' carregado para o modelo.")
    except FileNotFoundError:
        print(f"Erro critico: Arquivo '{caminho_arquivo_processado}' nao pode ser lido.")
        return

    dataframe_final = pd.get_dummies(dataframe_processado, columns=['Faixa_Etaria'], drop_first=True) # Converte a coluna 'Faixa_Etaria' em variáveis dummy (one-hot encoding) e remove a primeira coluna para evitar multicolinearidade
    X = dataframe_final.drop('Outcome', axis=1) # Remove a coluna 'Outcome' para obter as features 
    y = dataframe_final['Outcome'] # Obtém os rótulos do dataset 
    
    # Divisão dos dados em conjuntos de treino e teste através do método train_test_split 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) 
    print("\nDivisao dos dados em treino e teste concluida.") 

    criterios = ['information_gain', 'gain_ratio', 'gini']
    lista_resultados = []
    
    for criterio_atual in criterios:
        print(f"\n===== AVALIACAO DO MODELO COM CRITERIO: {criterio_atual.upper()} =====")
        arvore = ArvoreDeDecisao(profundidade_maxima=10, criterio=criterio_atual)
        arvore.fit(X_train, y_train)
        
        y_pred_treino = arvore.predict(X_train)
        print("\n> Resultados no Conjunto de TREINAMENTO:")
        print(f"  Acuracia: {accuracy_score(y_train, y_pred_treino):.4f}")
        print(f"  Precisao: {precision_score(y_train, y_pred_treino):.4f}")
        print(f"  Recall:   {recall_score(y_train, y_pred_treino):.4f}")
        print(f"  F1-Score: {f1_score(y_train, y_pred_treino):.4f}\n")
        print("  Matriz de Confusao (Treino):\n", confusion_matrix(y_train, y_pred_treino))

        y_pred_teste = arvore.predict(X_test)
        
        acc = accuracy_score(y_test, y_pred_teste)
        prec = precision_score(y_test, y_pred_teste)
        rec = recall_score(y_test, y_pred_teste)
        f1 = f1_score(y_test, y_pred_teste)
        lista_resultados.append({
            'Criterio': criterio_atual,
            'Acuracia': acc,
            'Precisao': prec,
            'Recall': rec,
            'F1-Score': f1
        })
        
        print("\n> Resultados no Conjunto de TESTE:")
        print(f"  Acuracia: {acc:.4f}")
        print(f"  Precisao: {prec:.4f}")
        print(f"  Recall:   {rec:.4f}")
        print(f"  F1-Score: {f1:.4f}\n")
        print("  Matriz de Confusao (Teste):\n", confusion_matrix(y_test, y_pred_teste))
        print("=" * 60)
        
    print("\n\n--- TABELA COMPARATIVA DE RESULTADOS (CONJUNTO DE TESTE) ---\n")
    dataframe_resultados = pd.DataFrame(lista_resultados) # Cria um DataFrame a partir da lista de resultados 
    dataframe_resultados.set_index('Criterio', inplace=True) # Define a coluna 'Criterio' como índice do DataFrame
    print(dataframe_resultados.round(4)) 
    
    print("\nGerando grafico comparativo dos modelos...")
    nome_arquivo_grafico = "comparacao_modelos.png"
    
    dataframe_resultados.plot(kind='bar', figsize=(14, 8), rot=0) # Cria um gráfico de barras comparando os resultados dos modelos
    plt.title('Comparacao de Performance dos Modelos no Conjunto de Teste', fontsize=16)
    plt.ylabel('Pontuacao (Score)', fontsize=12)
    plt.xlabel('Criterio de Divisao', fontsize=12)
    plt.ylim(0, 1.0)
    plt.legend(title='Metricas')
    plt.tight_layout()
    plt.savefig(nome_arquivo_grafico)
    
    print(f"Grafico comparativo salvo com sucesso como '{nome_arquivo_grafico}'")

if __name__ == '__main__':
    principal()