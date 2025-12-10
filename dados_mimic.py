# dados_mimic.py

from typing import List, Dict, Optional
import os

import pandas as pd
import config


def _carregar_rotulos_automaticos(
    path_diag: str,
    path_dic: str,
) -> Optional[pd.DataFrame]:
    """
    Lê DIAGNOSES_ICD*_filtred.csv e D_ICD_DIAGNOSES*_filtred.csv e constrói
    uma tabela HADM_ID -> DIAGNOSTICO_VERDADEIRO (LONG_TITLE do código
    principal, i.e. menor SEQ_NUM).
    """
    if not (os.path.exists(path_diag) and os.path.exists(path_dic)):
        return None

    # DIAGNOSES_ICD: ROW_ID,SUBJECT_ID,HADM_ID,SEQ_NUM,ICD9_CODE
    df_diag = pd.read_csv(path_diag, dtype=str)

    # garantir que SEQ_NUM é numérico para ordenar
    df_diag["SEQ_NUM_FLOAT"] = df_diag["SEQ_NUM"].astype(float)

    # escolher o diagnóstico principal por HADM_ID (menor SEQ_NUM)
    df_prim = (
        df_diag.sort_values("SEQ_NUM_FLOAT")
        .groupby("HADM_ID")
        .first()
        .reset_index()
    )
    df_prim = df_prim[["HADM_ID", "ICD9_CODE"]]

    # D_ICD_DIAGNOSES: ROW_ID,ICD9_CODE,SHORT_TITLE,LONG_TITLE
    df_dic = pd.read_csv(path_dic, dtype=str)

    df_prim = df_prim.merge(
        df_dic[["ICD9_CODE", "LONG_TITLE"]],
        on="ICD9_CODE",
        how="left",
    )

    df_prim.rename(columns={"LONG_TITLE": "DIAGNOSTICO_VERDADEIRO"}, inplace=True)

    return df_prim[["HADM_ID", "DIAGNOSTICO_VERDADEIRO"]]


def carregar_casos_mimic(
    path: str = None,
    n_max: Optional[int] = None,
) -> List[Dict]:
    """
    Lê o CSV de notas (NOTEEVENTS_random_separado_filtred.csv) e junta
    automaticamente o diagnóstico principal de cada admissão com base
    em DIAGNOSES_ICD_random_filtred.csv + D_ICD_DIAGNOSES_filtred.csv.

    Retorna uma lista de dicionários com:
      - id
      - subject_id
      - hadm_id
      - descricao
      - diagnostico_verdadeiro (LONG_TITLE) ou None se não houver
    """
    if path is None:
        path = config.CAMINHO_CASOS

    df_notes = pd.read_csv(path, dtype=str)

    # amostragem opcional
    if n_max is not None and n_max < len(df_notes):
        df_notes = df_notes.sample(n_max, random_state=42)

    # limpar/normalizar HADM_ID das notas (ex.: "174105.0" -> 174105)
    df_notes["HADM_ID_NORM"] = (
        df_notes["HADM_ID"]
        .astype(float)
        .astype("Int64")
        .astype(str)
    )

    # carregar rótulos automáticos
    df_rot = _carregar_rotulos_automaticos(
        config.CAMINHO_DIAGNOSES_ICD,
        config.CAMINHO_D_ICD_DIAGNOSES,
    )

    if df_rot is not None:
        df_rot["HADM_ID_NORM"] = df_rot["HADM_ID"].astype(str)
        df_merge = df_notes.merge(
            df_rot[["HADM_ID_NORM", "DIAGNOSTICO_VERDADEIRO"]],
            on="HADM_ID_NORM",
            how="left",
        )
    else:
        df_notes["DIAGNOSTICO_VERDADEIRO"] = None
        df_merge = df_notes

    casos: List[Dict] = []
    for idx, row in df_merge.iterrows():
        casos.append(
            {
                "id": int(idx),
                "subject_id": row["SUBJECT_ID"],
                "hadm_id": row["HADM_ID"],
                "descricao": row["NOTE_TEXT"],
                "diagnostico_verdadeiro": row.get("DIAGNOSTICO_VERDADEIRO"),
            }
        )
    return casos
