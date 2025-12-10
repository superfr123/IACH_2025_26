import os
import pandas as pd


# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ORIG = os.path.join(BASE_DIR, "data", "original")
DATA_OUT = os.path.join(BASE_DIR, "data")

# Ficheiros de entrada (ajusta os nomes se forem diferentes)
PATH_NOTES = os.path.join(DATA_ORIG, "NOTEEVENTS_random_separado.csv")
PATH_DIAG = os.path.join(DATA_ORIG, "DIAGNOSES_ICD_random.csv")
PATH_DIC = os.path.join(DATA_ORIG, "D_ICD_DIAGNOSES.csv")


def norm_hadm(x):
    """Normaliza HADM_ID para string de inteiro (ex.: '174105.0' -> '174105')."""
    if pd.isna(x) or x == "":
        return None
    try:
        return str(int(float(x)))
    except ValueError:
        return None


def main():
    # ----------------- Ler ficheiros -----------------
    print("A ler NOTEEVENTS_random_separado.csv...")
    notes = pd.read_csv(PATH_NOTES, dtype=str)

    print("A ler DIAGNOSES_ICD_random.csv...")
    diag = pd.read_csv(PATH_DIAG, dtype=str)

    print("A ler D_ICD_DIAGNOSES.csv...")
    dic_icd = pd.read_csv(PATH_DIC, dtype=str)

    # ----------------- Normalizar HADM_ID -----------------
    notes["HADM_ID_NORM"] = notes["HADM_ID"].apply(norm_hadm)
    diag["HADM_ID_NORM"] = diag["HADM_ID"].apply(norm_hadm)

    # remover linhas sem HADM_ID válido
    notes_valid = notes[notes["HADM_ID_NORM"].notna()].copy()
    diag_valid = diag[diag["HADM_ID_NORM"].notna()].copy()

    print(f"Notas válidas: {len(notes_valid)} / {len(notes)}")
    print(f"Diagnósticos válidos: {len(diag_valid)} / {len(diag)}")

    # ----------------- Chave combinada SUBJECT_ID + HADM_ID_NORM -----------------
    notes_valid["KEY"] = (
        notes_valid["SUBJECT_ID"].astype(str) + "|" + notes_valid["HADM_ID_NORM"].astype(str)
    )
    diag_valid["KEY"] = (
        diag_valid["SUBJECT_ID"].astype(str) + "|" + diag_valid["HADM_ID_NORM"].astype(str)
    )

    keys_notes = set(notes_valid["KEY"])
    keys_diag = set(diag_valid["KEY"])
    common_keys = keys_notes & keys_diag

    print(f"Admissões em comum entre notas e diagnósticos: {len(common_keys)}")

    # ----------------- Filtrar linhas pelas admissões em comum -----------------
    notes_filt = notes_valid[notes_valid["KEY"].isin(common_keys)].drop(
        columns=["HADM_ID_NORM", "KEY"]
    )
    diag_filt = diag_valid[diag_valid["KEY"].isin(common_keys)].drop(
        columns=["HADM_ID_NORM", "KEY"]
    )

    print(f"Notas depois do filtro: {len(notes_filt)}")
    print(f"Diagnósticos depois do filtro: {len(diag_filt)}")

    # ----------------- Filtrar dicionário ICD -----------------
    codigos_usados = diag_filt["ICD9_CODE"].dropna().unique()
    dic_filt = dic_icd[dic_icd["ICD9_CODE"].isin(codigos_usados)].copy()

    print(f"Códigos ICD usados: {len(codigos_usados)}")
    print(f"Entradas no dicionário depois do filtro: {len(dic_filt)}")

    # ----------------- Guardar saídas -----------------
    os.makedirs(DATA_OUT, exist_ok=True)

    out_notes = os.path.join(DATA_OUT, "NOTEEVENTS_random_separado_filtred.csv")
    out_diag = os.path.join(DATA_OUT, "DIAGNOSES_ICD_random_filtred.csv")
    out_dic = os.path.join(DATA_OUT, "D_ICD_DIAGNOSES_filtred.csv")

    notes_filt.to_csv(out_notes, index=False)
    diag_filt.to_csv(out_diag, index=False)
    dic_filt.to_csv(out_dic, index=False)

    print("\nFicheiros filtrados gravados em:")
    print(" -", out_notes)
    print(" -", out_diag)
    print(" -", out_dic)


if __name__ == "__main__":
    main()
