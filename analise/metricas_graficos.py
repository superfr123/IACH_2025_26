# analise_resultados.py
#
# Script para analisar o ficheiro:
#   output/historico_experimentos.csv
#
# Podes correr como script (python analise_resultados.py)
# ou copiar por blocos para um notebook Jupyter.

import os
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------
# 1) Configuração básica
# ---------------------------------------------------------

# Caminho para o ficheiro de histórico (ajusta se estiver noutro sítio)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
HIST_PATH = os.path.join(BASE_DIR, "output", "historico_experimentos.csv")

print("A carregar histórico de:", HIST_PATH)
df = pd.read_csv(HIST_PATH)

print("Número de linhas no histórico:", len(df))
print(df.head())


# ---------------------------------------------------------
# 2) Limpeza de tipos (booleans, tempos, etc.)
# ---------------------------------------------------------

def to_bool(x):
    """Converte strings 'True'/'False' em bool; devolve None se vazio."""
    if isinstance(x, bool):
        return x
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    if s in ["true", "1", "yes", "y"]:
        return True
    if s in ["false", "0", "no", "n"]:
        return False
    return None

df["acertou_A_bool"] = df["acertou_A"].apply(to_bool)
df["acertou_B_bool"] = df["acertou_B"].apply(to_bool)

# Converter tempos para float (caso tenham vindo como strings)
for col in ["tempo_total", "tempo_grafo", "tempo_medicos"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Converter reputações para float (podem ter NaN na 1ª iteração se não houve ground truth)
for col in ["reputacao_A", "reputacao_B"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# Garantir que discordancia e num_componentes são numéricos
df["discordancia"] = pd.to_numeric(df["discordancia"], errors="coerce")
df["num_componentes"] = pd.to_numeric(df["num_componentes"], errors="coerce")
df["len_nota"] = pd.to_numeric(df["len_nota"], errors="coerce")


# Filtrar só linhas com diagnóstico verdadeiro (para métricas de acerto)
df_gt = df[df["diagnostico_verdadeiro"].notna()].copy()
print("Casos com ground truth:", len(df_gt))


# ---------------------------------------------------------
# 3) Métricas de desempenho dos médicos
# ---------------------------------------------------------

def taxa_acerto(col_bool):
    serie = df_gt[col_bool].dropna()
    if len(serie) == 0:
        return None
    return serie.mean()

acc_A = taxa_acerto("acertou_A_bool")
acc_B = taxa_acerto("acertou_B_bool")

print("\n=== Taxa de acerto global ===")
print(f"Médico A: {acc_A:.3f}" if acc_A is not None else "Médico A: sem dados")
print(f"Médico B: {acc_B:.3f}" if acc_B is not None else "Médico B: sem dados")

# ---- Gráfico 1: barras com acerto A vs B ----
plt.figure()
plt.bar(["Médico A", "Médico B"], [acc_A, acc_B])
plt.ylim(0, 1)
plt.ylabel("Taxa de acerto")
plt.title("Taxa de acerto global dos médicos (casos com ground truth)")
plt.show()


# ---------------------------------------------------------
# 4) Evolução da reputação ao longo das iterações
# ---------------------------------------------------------

plt.figure()
plt.plot(df["iteracao"], df["reputacao_A"], marker="o", label="Médico A")
plt.plot(df["iteracao"], df["reputacao_B"], marker="o", label="Médico B")
plt.xlabel("Iteração")
plt.ylabel("Reputação")
plt.title("Evolução da reputação dos médicos")
plt.legend()
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# 5) Discordância entre médicos
# ---------------------------------------------------------

# Histograma de discordância
plt.figure()
df["discordancia"].plot(kind="hist", bins=10)
plt.xlabel("Discordância (0 = total acordo, 1 = total desacordo)")
plt.ylabel("Frequência")
plt.title("Distribuição da discordância entre médicos")
plt.show()

# Discordância vs falhas
# Definimos baixa discordância < 0.3, alta >= 0.3 (podes ajustar o limiar)
threshold = 0.3
df_gt["discordancia_alta"] = df_gt["discordancia"] >= threshold

# Casos em que pelo menos um médico falhou
df_gt["algum_falhou"] = (~df_gt["acertou_A_bool"].fillna(False)) | \
                        (~df_gt["acertou_B_bool"].fillna(False))

tab = pd.crosstab(df_gt["discordancia_alta"], df_gt["algum_falhou"], normalize="index")
print("\n=== Falhas vs discordância (linha normalizada) ===")
print(tab)

# Gráfico 2: barras – probabilidade de falha em baixa vs alta discordância
prob_falha_baixa = tab.loc[False, True] if False in tab.index and True in tab.columns else 0
prob_falha_alta = tab.loc[True, True]  if True in tab.index  and True in tab.columns else 0

plt.figure()
plt.bar(["Discordância baixa", "Discordância alta"], [prob_falha_baixa, prob_falha_alta])
plt.ylim(0, 1)
plt.ylabel("Probabilidade de pelo menos um médico falhar")
plt.title("Falhas vs nível de discordância")
plt.show()


# ---------------------------------------------------------
# 6) Relação com a estrutura do grafo (num_componentes)
# ---------------------------------------------------------

# Scatter: num_componentes vs discordância
plt.figure()
plt.scatter(df["num_componentes"], df["discordancia"])
plt.xlabel("Número de componentes desconectadas no grafo")
plt.ylabel("Discordância")
plt.title("Discordância vs número de componentes do grafo")
plt.grid(True)
plt.show()

# Opcional: taxa de acerto por grupo de num_componentes (<=1 vs >1)
df_gt["grafo_fragmentado"] = df_gt["num_componentes"] > 1

acc_A_simples = df_gt[~df_gt["grafo_fragmentado"]]["acertou_A_bool"].mean()
acc_A_frag = df_gt[df_gt["grafo_fragmentado"]]["acertou_A_bool"].mean()
acc_B_simples = df_gt[~df_gt["grafo_fragmentado"]]["acertou_B_bool"].mean()
acc_B_frag = df_gt[df_gt["grafo_fragmentado"]]["acertou_B_bool"].mean()

plt.figure()
labels = ["A simples", "A fragmentado", "B simples", "B fragmentado"]
vals = [acc_A_simples, acc_A_frag, acc_B_simples, acc_B_frag]
plt.bar(labels, vals)
plt.ylim(0, 1)
plt.ylabel("Taxa de acerto")
plt.title("Taxa de acerto vs fragmentação do grafo")
plt.xticks(rotation=20)
plt.show()


# ---------------------------------------------------------
# 7) Tempos de execução
# ---------------------------------------------------------

print("\n=== Tempos médios (s) ===")
for col in ["tempo_grafo", "tempo_medicos", "tempo_total"]:
    media = df[col].mean()
    std = df[col].std()
    print(f"{col}: média = {media:.2f} s, desvio-padrão = {std:.2f} s")

# Gráfico 3: barras com tempos médios
plt.figure()
means = [df["tempo_grafo"].mean(), df["tempo_medicos"].mean(), df["tempo_total"].mean()]
plt.bar(["Grafo", "Médicos", "Total"], means)
plt.ylabel("Tempo médio (s)")
plt.title("Tempos médios por caso")
plt.show()

# Scatter: tamanho da nota vs tempo total
plt.figure()
plt.scatter(df["len_nota"], df["tempo_total"])
plt.xlabel("Comprimento da nota (nº de caracteres)")
plt.ylabel("Tempo total (s)")
plt.title("Tempo total vs comprimento da nota")
plt.grid(True)
plt.show()


# ---------------------------------------------------------
# 8) (Opcional) Guardar um pequeno resumo em texto
# ---------------------------------------------------------

resumo_path = os.path.join(BASE_DIR, "output", "resumo_metricas.txt")
with open(resumo_path, "w", encoding="utf-8") as f:
    f.write("Resumo de métricas (gerado por analise_resultados.py)\n\n")
    f.write(f"Número total de casos no histórico: {len(df)}\n")
    f.write(f"Casos com ground truth: {len(df_gt)}\n\n")
    f.write("Taxa de acerto global:\n")
    f.write(f"  Médico A: {acc_A:.3f}\n")
    f.write(f"  Médico B: {acc_B:.3f}\n\n")
    for col in ["tempo_grafo", "tempo_medicos", "tempo_total"]:
        media = df[col].mean()
        std = df[col].std()
        f.write(f"{col}: média = {media:.2f} s, desvio-padrão = {std:.2f} s\n")

print("\nResumo de métricas guardado em:", resumo_path)
