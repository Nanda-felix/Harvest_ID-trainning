# Projeto de Classificação de Imagens

Este projeto contém três modelos de classificação de imagens que podem ser treinados usando scripts Python. Este documento explica como configurar o ambiente, instalar as dependências necessárias e treinar os modelos.

## Requisitos

Antes de começar, certifique-se de ter o seguinte instalado:

- Python 3.7 ou superior
- pip (gerenciador de pacotes do Python)
- Um ambiente de desenvolvimento adequado (como VSCode, PyCharm, etc.)

## Instalação

1. **Clone o repositório:**

   ```bash
   git clone https://github.com/Nanda-felix/Harvest_ID-trainning.git
   cd Harvest_ID-trainning
   
2. **Instale as dependências:**

    ```bash
    pip install -r requirements.txt
    ```
## Treinamento dos Modelos

  Os três modelos devem ser treinados um após o outro na ordem indicada abaixo. Certifique-se de aguardar a conclusão de cada treinamento antes de iniciar o próximo.

 1. **Treine o modelo "Planta ou Não":**

    ```bash
    python Training_Codes/treinamento_planta_ou_nao.py
     ```

2. **Treine o modelo "Bandeiras":**
     ```bash
    python Training_Codes/treinamento_bandeiras.py
     ```
     
3. **Treine o modelo "Tomato":**
    ```bash
   python Training_Codes/treinamento_tomato.py
     ```

