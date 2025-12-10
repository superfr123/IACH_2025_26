# avaliacao.py

from typing import List
from difflib import SequenceMatcher


def _similaridade(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()


def diagnostico_correto(
    diagnosticos_modelo: List[str],
    diagnostico_verdadeiro: str,
    limiar: float = 0.6,
) -> bool:
    """
    Considera correto se algum dos diagnósticos do modelo for
    suficientemente parecido (em termos de string) ao diagnóstico
    verdadeiro.

    - diagnosticos_modelo: lista de nomes de diagnósticos previstos pelo modelo
    - diagnostico_verdadeiro: string com o rótulo verdadeiro
      (por ex. "congestive heart failure")

    O limiar de similaridade pode ser ajustado (default 0.6).
    """
    if not diagnostico_verdadeiro:
        return False

    alvo = diagnostico_verdadeiro.strip().lower()
    candidatos = [d.strip().lower() for d in diagnosticos_modelo if d]

    for cand in candidatos:
        if _similaridade(alvo, cand) >= limiar:
            return True
    return False
