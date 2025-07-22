# =============================================================================
# TRABALHO DE INTELIGÊNCIA ARTIFICIAL - ÁRVORE DE DECISÃO
#
# Este script realiza todas as etapas do trabalho:
# 1. Cria um dataset pré-processado e o salva em um arquivo CSV.
# 2. Implementa o algoritmo de Árvore de Decisão do zero.
# 3. Treina três versões do modelo (Information Gain, Gain Ratio, Gini).
# 4. Avalia os modelos e exibe as métricas de performance.
# =============================================================================

import pandas as pd
import numpy as np
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import os # Usado para verificar se o arquivo já existe

# -------------------------------------------------------------------
# SEÇÃO 1: CRIAÇÃO E PREPARAÇÃO DO DATASET
# -------------------------------------------------------------------

def criar_dataset_processado():
    """
    Carrega o dataset Pima Indians original, aplica o pré-processamento
    e salva o resultado em um novo arquivo CSV.
    """
    nome_arquivo_saida = "pima_diabetes_processado.csv"
    
    # Verifica se o arquivo já foi criado para não refazer o trabalho
    if os.path.exists(nome_arquivo_saida):
        print(f"O arquivo '{nome_arquivo_saida}' já existe. Pulando a etapa de criação.")
        return

    print("Iniciando a criação do dataset pré-processado...")

    # Carregar o dataset original da web
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/pima-indians-diabetes/pima-indians-diabetes.data"
    colunas = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    df = pd.read_csv(url, header=None, names=colunas)

    # 1. Tratamento de dados faltantes (zeros) com a mediana
    cols_com_zeros_invalidos = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for coluna in cols_com_zeros_invalidos:
        df[coluna] = df[coluna].replace(0, np.nan)
        mediana = df[coluna].median()
        df[coluna].fillna(mediana, inplace=True)
    
    # 2. Categorização do atributo 'Age'
    bins = [0, 29, 59, 150]
    labels = ['Jovem', 'Adulto', 'Idoso']
    df['Faixa_Etaria'] = pd.cut(df['Age'], bins=bins, labels=labels, right=True)

    # Salvar o novo dataset em um arquivo CSV
    df.to_csv(nome_arquivo_saida, index=False)
    print(f"Dataset pré-processado foi salvo com sucesso como '{nome_arquivo_saida}'")


# -------------------------------------------------------------------
# SEÇÃO 2: FUNÇÕES DE CÁLCULO DAS MÉTRICAS DE DIVISÃO
# -------------------------------------------------------------------

def calcular_entropia(y):
    contagem_classes = np.bincount(y)
    probabilidades = contagem_classes / len(y)
    probabilidades = probabilidades[probabilidades > 0]
    return -np.sum(probabilidades * np.log2(probabilidades))

def calcular_information_gain(y_pai, y_esquerda, y_direita):
    peso_esquerda = len(y_esquerda) / len(y_pai)
    peso_direita = len(y_direita) / len(y_pai)
    entropia_pai = calcular_entropia(y_pai)
    entropia_ponderada_filhos = (peso_esquerda * calcular_entropia(y_esquerda) +
                                 peso_direita * calcular_entropia(y_direita))
    return entropia_pai - entropia_ponderada_filhos

def calcular_gini(y):
    contagem_classes = np.bincount(y)
    probabilidades = contagem_classes / len(y)
    return 1 - np.sum(probabilidades**2)

def calcular_gini_gain(y_pai, y_esquerda, y_direita):
    peso_esquerda = len(y_esquerda) / len(y_pai)
    peso_direita = len(y_direita) / len(y_pai)
    gini_pai = calcular_gini(y_pai)
    gini_ponderado_filhos = (peso_esquerda * calcular_gini(y_esquerda) +
                             peso_direita * calcular_gini(y_direita))
    return gini_pai - gini_ponderado_filhos

def calcular_gain_ratio(y_pai, y_esquerda, y_direita):
    info_gain = calcular_information_gain(y_pai, y_esquerda, y_direita)
    if info_gain == 0: return 0
    proporcao_esquerda = len(y_esquerda) / len(y_pai)
    proporcao_direita = len(y_direita) / len(y_pai)
    split_info = - (proporcao_esquerda * np.log2(proporcao_esquerda + 1e-9) +
                    proporcao_direita * np.log2(proporcao_direita + 1e-9))
    if split_info == 0: return 0
    return info_gain / split_info


# -------------------------------------------------------------------
# SEÇÃO 3: IMPLEMENTAÇÃO DA ÁRVORE DE DECISÃO
# -------------------------------------------------------------------

class Node:
    """Representa um nó na árvore de decisão."""
    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        return self.value is not None

class DecisionTree:
    """Implementação do algoritmo de Árvore de Decisão do zero."""
    def __init__(self, min_samples_split=2, max_depth=100, criterion='information_gain'):
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.criterion = criterion
        self.root = None

    def fit(self, X, y):
        # Converte para NumPy arrays para performance
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        y_vals = y.values if isinstance(y, pd.Series) else y
        self.root = self._build_tree(X_vals, y_vals)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_labels = len(np.unique(y))

        # Condições de parada
        if (depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split):
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        feat_idxs = np.random.choice(n_features, n_features, replace=False)
        best_split = self._find_best_split(X, y, feat_idxs)
        
        if not best_split or best_split['gain'] <= 0:
            leaf_value = self._most_common_label(y)
            return Node(value=leaf_value)

        left_idxs, right_idxs = best_split['indices']
        left_tree = self._build_tree(X[left_idxs, :], y[left_idxs], depth + 1)
        right_tree = self._build_tree(X[right_idxs, :], y[right_idxs], depth + 1)
        
        return Node(best_split['feature'], best_split['threshold'], left_tree, right_tree)

    def _find_best_split(self, X, y, feat_idxs):
        best_gain = -1
        best_split = None

        for feat_idx in feat_idxs:
            thresholds = np.unique(X[:, feat_idx])
            for threshold in thresholds:
                left_idxs = np.where(X[:, feat_idx] <= threshold)[0]
                right_idxs = np.where(X[:, feat_idx] > threshold)[0]

                if len(left_idxs) == 0 or len(right_idxs) == 0: continue

                # Seleciona a função de ganho baseada no critério
                if self.criterion == 'information_gain':
                    gain = calcular_information_gain(y, y[left_idxs], y[right_idxs])
                elif self.criterion == 'gain_ratio':
                    gain = calcular_gain_ratio(y, y[left_idxs], y[right_idxs])
                else: # 'gini'
                    gain = calcular_gini_gain(y, y[left_idxs], y[right_idxs])

                if gain > best_gain:
                    best_gain = gain
                    best_split = {'feature': feat_idx, 'threshold': threshold,
                                  'indices': (left_idxs, right_idxs), 'gain': gain}
        return best_split

    def _most_common_label(self, y):
        return Counter(y).most_common(1)[0][0]

    def predict(self, X):
        X_vals = X.values if isinstance(X, pd.DataFrame) else X
        return np.array([self._traverse_tree(x, self.root) for x in X_vals])

    def _traverse_tree(self, x, node):
        if node.is_leaf_node(): return node.value
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        return self._traverse_tree(x, node.right)


# -------------------------------------------------------------------
# SEÇÃO 4: EXECUÇÃO PRINCIPAL
# -------------------------------------------------------------------

def main():
    """Função principal que orquestra todo o processo."""
    
    # Passo 1: Gerar o arquivo de dados pré-processado
    criar_dataset_processado()

    # Passo 2: Carregar o dataset do arquivo
    try:
        df_processado = pd.read_csv("pima_diabetes_processado.csv")
        print("\nDataset 'pima_diabetes_processado.csv' carregado para o modelo.")
    except FileNotFoundError:
        print("Erro crítico: Arquivo 'pima_diabetes_processado.csv' não pôde ser lido.")
        return

    # Passo 3: Preparar os dados para o modelo (One-Hot Encoding)
    df_final = pd.get_dummies(df_processado, columns=['Faixa_Etaria'], drop_first=True)
    X = df_final.drop('Outcome', axis=1)
    y = df_final['Outcome']
    
    # Passo 4: Divisão dos Dados em Treino e Teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Passo 5: Treinar e Avaliar os três modelos
    criterios = ['information_gain', 'gain_ratio', 'gini']
    
    for crit in criterios:
        print(f"\n===== AVALIAÇÃO DO MODELO COM CRITÉRIO: {crit.upper()} =====")
        
        tree = DecisionTree(max_depth=10, criterion=crit)
        tree.fit(X_train, y_train)
        
        # Avaliação no conjunto de TREINAMENTO
        y_pred_train = tree.predict(X_train)
        print("\n> Resultados no Conjunto de TREINAMENTO:")
        print(f"  Acurácia: {accuracy_score(y_train, y_pred_train):.4f}")
        print(f"  Precisão: {precision_score(y_train, y_pred_train):.4f}")
        print(f"  Recall:   {recall_score(y_train, y_pred_train):.4f}")
        print(f"  F1-Score: {f1_score(y_train, y_pred_train):.4f}")
        print("  Matriz de Confusão (Treino):\n", confusion_matrix(y_train, y_pred_train))
        
        # Avaliação no conjunto de TESTE
        y_pred_test = tree.predict(X_test)
        print("\n> Resultados no Conjunto de TESTE:")
        print(f"  Acurácia: {accuracy_score(y_test, y_pred_test):.4f}")
        print(f"  Precisão: {precision_score(y_test, y_pred_test):.4f}")
        print(f"  Recall:   {recall_score(y_test, y_pred_test):.4f}")
        print(f"  F1-Score: {f1_score(y_test, y_pred_test):.4f}")
        print("  Matriz de Confusão (Teste):\n", confusion_matrix(y_test, y_pred_test))
        print("=" * 55)

# Ponto de entrada do script
if __name__ == '__main__':
    main()