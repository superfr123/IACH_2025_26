# IACH_2025_26 - Médicos virtuais com grafo de conhecimento e reputação (MIMIC-III)

Projeto para o **Practical Lab de Human-Centred AI (HcAI)**.

Primeiramente é preciso ter uma chave API do openai que qualquer um pode obter com a sua conta do chatgpt.
É preciso que coloquem essa chave no config.py no sitio devidamente sinalizado entre aspas (para ser uma string).
Para executar o programa basta executar o ficheiro python main.py.

O objetivo é explorar um sistema de apoio ao diagnóstico baseado em **grandes modelos de linguagem (LLMs)** que:

1. Recebe **notas clínicas reais** do MIMIC-III;
2. Constrói um **grafo de conhecimento clínico** a partir da nota;
3. Usa **dois médicos virtuais** (agentes LLM com perfis diferentes) para propor diagnósticos;
4. Compara as respostas com o **diagnóstico verdadeiro (ICD-9)**;
5. Atualiza uma **reputação** por médico e mede a **discordância** entre eles;
6. Regista tudo num CSV para análise posterior (métricas + gráficos).

source:
https://www.kaggle.com/datasets/bilal1907/mimic-iii-10k
