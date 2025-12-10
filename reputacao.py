# reputacao.py

from dataclasses import dataclass, asdict
from typing import Dict
import json
import os

import config  # novo import


@dataclass
class EstatisticasMedico:
    acertos: int = 0
    total: int = 0

    @property
    def reputacao(self) -> float:
        # prior neutro 0.5 quando ainda não há dados
        if self.total == 0:
            return 0.5
        return self.acertos / self.total


class GestorReputacao:
    def __init__(self, medico_ids, path_json: str | None = None):
        if path_json is None:
            # reputação em output/reputacao.json dentro do proj/
            path_json = os.path.join(config.OUTPUT_DIR, "reputacao.json")

        self.path_json = path_json
        self.medicos: Dict[str, EstatisticasMedico] = {
            mid: EstatisticasMedico() for mid in medico_ids
        }

        # se já existir ficheiro, carregar; caso contrário, criar de raiz
        self._carregar_se_existir()
        if not os.path.exists(self.path_json):
            self._guardar()

    def _carregar_se_existir(self):
        if os.path.exists(self.path_json):
            with open(self.path_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            for mid, stats in data.items():
                self.medicos[mid] = EstatisticasMedico(**stats)

    def _guardar(self):
        os.makedirs(os.path.dirname(self.path_json), exist_ok=True)
        data = {mid: asdict(stats) for mid, stats in self.medicos.items()}
        with open(self.path_json, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def atualizar(self, medico_id: str, correto: bool):
        stats = self.medicos[medico_id]
        stats.total += 1
        if correto:
            stats.acertos += 1
        self._guardar()

    def obter_reputacao(self, medico_id: str) -> float:
        return self.medicos[medico_id].reputacao

    def melhor_medico(self) -> str:
        return max(self.medicos.keys(), key=lambda mid: self.medicos[mid].reputacao)
