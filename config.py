# config.py

import os

# Diretório base = pasta onde está este ficheiro (proj/)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Pastas
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DIR_GRAFOS = os.path.join(OUTPUT_DIR, "grafos")

# Ficheiros de dados (já filtrados pelo preprocess_mimic.py)
CAMINHO_CASOS = os.path.join(DATA_DIR, "NOTEEVENTS_random_separado_filtred.csv")
CAMINHO_DIAGNOSES_ICD = os.path.join(DATA_DIR, "DIAGNOSES_ICD_random_filtred.csv")
CAMINHO_D_ICD_DIAGNOSES = os.path.join(DATA_DIR, "D_ICD_DIAGNOSES_filtred.csv")

# Modelo e parâmetros
MODEL_NAME = "gpt-4o-mini"
NUM_ITERACOES = 15
NUM_CASOS = 20

#tem de ter uma chave open ai valida e enviar:
#$env:OPENAI_API_KEY = "chave_aqui"
