# active_learning.py

from typing import List, Dict, Any
import random


def escolher_proximo_caso_random(indices_restantes: List[int]) -> int:
    return random.choice(indices_restantes)


def calcular_discordancia(
    diag_medico_a: List[Dict[str, Any]],
    diag_medico_b: List[Dict[str, Any]],
) -> float:
    """
    Mede discordância entre dois médicos com base nos nomes dos diagnósticos.

    Usa 1 - Jaccard(similaridade entre conjuntos de nomes):
        J = |A ∩ B| / |A ∪ B|
        discordância = 1 - J
    """
    nomes_a = {d.get("name", "").strip().lower() for d in diag_medico_a if d.get("name")}
    nomes_b = {d.get("name", "").strip().lower() for d in diag_medico_b if d.get("name")}

    union = nomes_a | nomes_b
    if not union:
        return 0.0
    inter = nomes_a & nomes_b
    jaccard = len(inter) / len(union)
    return 1.0 - jaccard


def escolher_proximo_por_discordancia(
    casos: List[Dict[str, Any]],
    historico: List[Dict[str, Any]],
    indices_restantes: List[int],
) -> int:
    """
    Estratégia de active learning muito simples:

    - Se ainda não há histórico, escolhe um caso ao acaso.
    - Se já há histórico, identifica casos anteriores onde a discordância foi alta
      (por ex. >= 0.3) e calcula o comprimento médio das notas desses casos.
    - Depois escolhe, entre os índices_restantes, o caso cujo comprimento da nota
      é mais próximo desse comprimento médio (proxy de "casos semelhantes" em
      termos de complexidade do texto).

    É uma heurística barata que já ilustra o conceito de active learning:
    focar a anotação em regiões do espaço de dados onde os modelos discordam.
    """
    if not indices_restantes:
        raise ValueError("Sem índices restantes para escolher.")

    if not historico:
        return escolher_proximo_caso_random(indices_restantes)

    # filtrar casos com maior discordância
    altos = [h for h in historico if h.get("discordancia", 0.0) >= 0.3]
    if not altos:
        return escolher_proximo_caso_random(indices_restantes)

    media_len = sum(h["len_nota"] for h in altos) / len(altos)

    # escolher o índice cujo comprimento de nota é mais próximo da média dos casos "difíceis"
    melhor_idx = None
    melhor_diff = None
    for idx in indices_restantes:
        nota = casos[idx]["descricao"]
        l = len(str(nota))
        diff = abs(l - media_len)
        if melhor_diff is None or diff < melhor_diff:
            melhor_diff = diff
            melhor_idx = idx

    return melhor_idx if melhor_idx is not None else escolher_proximo_caso_random(indices_restantes)
