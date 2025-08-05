🌳 Projeto de Árvore de Decisão - Classificação de Diabetes
Este projeto consiste na implementação de um algoritmo de Árvore de Decisão do zero em Python para resolver um problema de classificação. O objetivo é prever o diagnóstico de diabetes 🩸 com base no dataset "Pima Indians Diabetes Database".

🛠️ Pré-requisitos
Python 3.8 ou superior instalado.

O gerenciador de pacotes pip (geralmente instalado junto com o Python).

📁 Estrutura de Arquivos
Para que o script funcione corretamente, o arquivo do dataset original deve estar na mesma pasta que o script Python. A estrutura deve ser a seguinte:

/SeuProjeto/

|-- AD.py             (ou o nome do script Python)

|-- diabetes.csv      (dataset original)

🚀 Instruções de Instalação e Execução
Como Python é uma linguagem interpretada, não há um passo de "compilação". O processo consiste em preparar o ambiente e instalar as bibliotecas necessárias.

1️⃣ Passo 1: Navegar até a Pasta do Projeto
Use o comando cd para entrar na pasta onde estão os seus arquivos.

cd caminho/para/a/pasta/do/projeto
Exemplo: cd C:\Users\palom\MachineLearning\ArvoreDecisao

2️⃣ Passo 2: Criar um Ambiente Virtual
Isso cria um ambiente Python isolado para o seu projeto, o que é uma boa prática.

python -m venv venv

3️⃣ Passo 3: Ativar o Ambiente Virtual

# No Windows (PowerShell)
.\venv\Scripts\Activate

4️⃣ Passo 4: Instalar as Dependências

pip install pandas numpy scikit-learn matplotlib seaborn

5️⃣ Passo 5: Executar o Script

python AD.py

📊 O que Esperar da Execução
Ao ser executado, o script irá:

Processar o arquivo diabetes.csv 🔄.

Salvar um novo arquivo limpo chamado diabetes_processado.csv 💾.

Exibir no terminal a análise de performance para os três modelos 💻.

Mostrar uma tabela comparativa final com os resultados dos testes.

Salvar um gráfico de barras com a comparação visual dos modelos no arquivo comparacao_modelos.png 🖼️.
