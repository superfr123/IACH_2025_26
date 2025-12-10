# main.py

import os
import csv
import time
import math  # <-- para testar NaN

from dados_mimic import carregar_casos_mimic
from grafo_conhecimento import ConstrutorGrafoLLM
from medicos import MedicoLLM, PROMPT_MEDICO_CONSERVADOR, PROMPT_MEDICO_EXPLORADOR
from reputacao import GestorReputacao
from avaliacao import diagnostico_correto
from active_learning import calcular_discordancia
import config


def _preparar_ficheiro_historico(path_csv: str):
    """Cria ficheiro de histórico com cabeçalho se ainda não existir."""
    os.makedirs(os.path.dirname(path_csv), exist_ok=True)
    if not os.path.exists(path_csv):
        with open(path_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    "iteracao",
                    "idx",
                    "subject_id",
                    "hadm_id",
                    "len_nota",
                    "diagnostico_verdadeiro",
                    "diag_A",
                    "diag_B",
                    "acertou_A",
                    "acertou_B",
                    "reputacao_A",
                    "reputacao_B",
                    "discordancia",
                    "num_componentes",
                    "tempo_total",
                    "tempo_grafo",
                    "tempo_medicos",
                ]
            )


def _tem_diagnostico_valido(diag):
    """Devolve True se o diagnóstico verdadeiro for uma string não vazia."""
    if diag is None:
        return False
    if isinstance(diag, float) and math.isnan(diag):
        return False
    s = str(diag).strip()
    return len(s) > 0


def main():
    # 1) Carregar casos (já com diagnostico_verdadeiro se houver)
    casos = carregar_casos_mimic(
        path=config.CAMINHO_CASOS,
        n_max=config.NUM_CASOS or None,
    )

    if not casos:
        print("Nenhum caso carregado. Verifica os ficheiros filtrados.")
        return

    # 2) Preparar construtor de grafo e médicos
    construtor_grafo = ConstrutorGrafoLLM(model_name=config.MODEL_NAME)

    medicos = {
        "A": MedicoLLM(
            "A",
            PROMPT_MEDICO_CONSERVADOR,
            model_name=config.MODEL_NAME,
            temperature=0.0,  # conservador, determinístico
        ),
        "B": MedicoLLM(
            "B",
            PROMPT_MEDICO_EXPLORADOR,
            model_name=config.MODEL_NAME,
            temperature=0.4,  # explorador, mais variabilidade
        ),
    }

    gestor_rep = GestorReputacao(medicos.keys())

    # 3) Preparar histórico em CSV
    historico_csv = os.path.join(config.OUTPUT_DIR, "historico_experimentos.csv")
    _preparar_ficheiro_historico(historico_csv)

    # 4) Iterar sobre casos (por agora em ordem; se quiseres usas active learning depois)
    indices_restantes = list(range(len(casos)))
    num_iter = min(config.NUM_ITERACOES, len(indices_restantes))

    for it in range(1, num_iter + 1):
        idx = indices_restantes.pop(0)
        caso = casos[idx]
        nota = caso["descricao"]
        diag_verdadeiro = caso.get("diagnostico_verdadeiro")

        # Verificar se existe diagnóstico verdadeiro válido; se não, saltar caso
        if not _tem_diagnostico_valido(diag_verdadeiro):
            print(
                f"\n=== Iteração {it} | SUBJECT {caso['subject_id']} "
                f"HADM {caso['hadm_id']} (idx {idx}) ==="
            )
            print("Sem diagnóstico verdadeiro (MIMIC). Caso ignorado na avaliação.\n")
            # não construímos grafo, não chamamos médicos, não registamos no CSV
            # também não fazemos sleep porque não houve chamadas ao modelo nesta iteração
            continue

        print(
            f"\n=== Iteração {it} | SUBJECT {caso['subject_id']} "
            f"HADM {caso['hadm_id']} (idx {idx}) ==="
        )
        print(f"Diagnóstico verdadeiro (MIMIC): {diag_verdadeiro}")

        # ------ início do timer total ------
        t_total_ini = time.perf_counter()

        # 4.1) Construir grafo (medir tempo do grafo)
        t_grafo_ini = time.perf_counter()
        grafo_res = construtor_grafo.construir(
            nota,
            output_dir=config.DIR_GRAFOS,
            nome_base=f"grafo_{caso['subject_id']}_{caso['hadm_id']}",
        )
        t_grafo_fim = time.perf_counter()
        tempo_grafo = t_grafo_fim - t_grafo_ini

        print(f"Grafo criado em: {grafo_res.html_path}")
        print(f"Número de componentes desconectadas no grafo: {grafo_res.num_componentes}")

        # 4.2) Diagnósticos dos médicos + reputação (medir tempo dos médicos)
        t_med_ini = time.perf_counter()

        resultados = {}
        nomes_diags_por_medico = {}
        acertou_por_medico = {"A": None, "B": None}
        reputacao_por_medico = {"A": None, "B": None}

        for mid, medico in medicos.items():
            res_med = medico.diagnosticar(nota, grafo_res.grafo_json)
            resultados[mid] = res_med

            nomes_diags = [d.get("name", "") for d in res_med.diagnoses]
            nomes_diags_por_medico[mid] = nomes_diags

            print(f"\nMédico {mid}: {nomes_diags}")

            correto = diagnostico_correto(nomes_diags, diag_verdadeiro)
            acertou_por_medico[mid] = correto
            gestor_rep.atualizar(mid, correto)
            rep = gestor_rep.obter_reputacao(mid)
            reputacao_por_medico[mid] = rep
            print(f"  -> {'ACERTOU' if correto else 'FALHOU'} (reputação = {rep:.2f})")

        t_med_fim = time.perf_counter()
        tempo_medicos = t_med_fim - t_med_ini

        # 4.3) Discordância entre médicos
        if "A" in resultados and "B" in resultados:
            discordancia = calcular_discordancia(
                resultados["A"].diagnoses,
                resultados["B"].diagnoses,
            )
        else:
            discordancia = 0.0

        print(f"\nDiscordância entre médicos (Jaccard-based): {discordancia:.2f}")

        # fim do timer total
        t_total_fim = time.perf_counter()
        tempo_total = t_total_fim - t_total_ini

        print(f"\nTempo grafo: {tempo_grafo:.2f} s")
        print(f"Tempo médicos: {tempo_medicos:.2f} s")
        print(f"Tempo total por caso: {tempo_total:.2f} s")

        # 4.4) Guardar no CSV de histórico (apenas para casos com ground truth válido)
        len_nota = len(str(nota))
        diag_A_str = "|".join(nomes_diags_por_medico.get("A", []))
        diag_B_str = "|".join(nomes_diags_por_medico.get("B", []))

        with open(historico_csv, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    it,
                    idx,
                    caso["subject_id"],
                    caso["hadm_id"],
                    len_nota,
                    diag_verdadeiro,
                    diag_A_str,
                    diag_B_str,
                    acertou_por_medico["A"],
                    acertou_por_medico["B"],
                    reputacao_por_medico["A"],
                    reputacao_por_medico["B"],
                    discordancia,
                    grafo_res.num_componentes,
                    tempo_total,
                    tempo_grafo,
                    tempo_medicos,
                ]
            )

        # 4.5) Pausa para não ultrapassar os 3 RPM (3 pedidos/iteração)
        if it < num_iter:
            alvo_segundos = 60.0  # queremos ~1 iteração por minuto
            espera = max(0.0, alvo_segundos - tempo_total)
            if espera > 0:
                print(f"A aguardar {espera:.1f} segundos para respeitar limite de RPM...\n")
                time.sleep(espera)

    print("\nFim da simulação.")
    print(f"Histórico de experiências guardado em: {historico_csv}")


if __name__ == "__main__":
    main()
