# 🌳 Projeto de Árvore de Decisão - Classificação de Diabetes

Este projeto implementa um algoritmo de Árvore de Decisão **do zero** em Python para resolver um problema de **classificação médica**.  
O objetivo é prever o diagnóstico de diabetes com base no dataset **Pima Indians Diabetes Database**.

## 🛠️ Pré-requisitos

- Python **3.8 ou superior**
- `pip` (gerenciador de pacotes do Python)

## 📁 Estrutura de Arquivos

Para que o script funcione corretamente, a estrutura da pasta deve ser:

/SeuProjeto/
│

├── AD.py # Script principal (algoritmo de decisão)

└── diabetes.csv # Dataset original

## 🚀 Instruções de Instalação e Execução

Como Python é uma linguagem interpretada, não há um passo de compilação.  
A seguir, o passo a passo para preparar o ambiente:

1️⃣ Navegar até a pasta do projeto

cd caminho/para/a/pasta/do/projeto

2️⃣ Criar um ambiente virtual 

python -m venv venv

3️⃣ Ativar o ambiente virtual

Windows (PowerShell): .\venv\Scripts\Activate

Linux/macOS: source venv/bin/activate

4️⃣ Instalar as dependências

pip install pandas numpy scikit-learn matplotlib seaborn

5️⃣ Executar o script

python AD.py

📊 O que Esperar da Execução
Ao executar o script, ele irá:

Processar o arquivo diabetes.csv 🔄

Gerar um novo arquivo limpo: diabetes_processado.csv 💾

Exibir no terminal a análise de performance de três modelos 💻

Mostrar uma tabela comparativa final com os resultados dos testes

Salvar um gráfico de barras no arquivo: comparacao_modelos.png 🖼️

📌 Observações
O projeto não depende de bibliotecas externas de Machine Learning para a construção da árvore de decisão – o algoritmo é implementado manualmente.

O dataset é amplamente utilizado em estudos de ML para classificação binária (diabetes: positivo/negativo).

📚 Dataset
Pima Indians Diabetes Database - Kaggle




