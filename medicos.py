# medicos.py

from dataclasses import dataclass
from typing import List, Dict, Any
import json

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


# -------------------------------------------------------------------
# PROMPTS DOS MÉDICOS
# -------------------------------------------------------------------

PROMPT_MEDICO_BASE = """
You are a medical doctor reasoning about a hospitalised patient.

You receive two inputs:
1) The raw clinical note, written in natural language.
2) A structured knowledge graph extracted from that note, represented as JSON
   with a list of nodes and a list of edges.

The knowledge graph contains nodes of different types, such as:
- patient
- demographic
- symptom
- sign
- risk_factor
- habit
- comorbidity
- test
- treatment
- intermediate_hypothesis    (e.g., "respiratory infection", "heart failure syndrome")
- diagnosis                  (e.g., "pneumonia", "congestive heart failure")
- other

Edges encode relations between nodes, such as:
- has_symptom, has_sign
- has_risk_factor, has_comorbidity
- received_test, received_treatment
- suggests_hypothesis
- supports_diagnosis
- suggests_diagnosis
- worsens, part_of, compatible_with, supports

Your task is to infer the most likely medical diagnoses for the CURRENT ADMISSION,
using BOTH the clinical note and the structure of the knowledge graph.

Pay special attention to:
- which symptoms and signs cluster together;
- which risk factors and comorbidities are present;
- which intermediate_hypothesis nodes support which diagnosis nodes;
- how tests, treatments and temporal evolution reinforce or weaken each hypothesis.

OUTPUT FORMAT

You must respond with a SINGLE JSON object encoded as text, with a top-level key "diagnoses".
The value of "diagnoses" must be a list of objects.
Each object in the list must contain:
- "name": the name of the diagnosis as a short string.
- "probability": a number between 0 and 1 expressing how likely this diagnosis is,
                 given the note and the graph.
- "justification": a short free-text explanation that explicitly mentions the key
                   symptoms, risk factors, comorbidities, tests and/or intermediate
                   hypotheses that support this diagnosis.

Guidelines:
- Include between 1 and 6 diagnoses, ordered from most to least likely.
- Probabilities do not need to sum to 1, but they must be coherent:
  the first diagnosis should have the highest probability.
- Prefer realistic, clinically coherent diagnoses that fit the provided data.
- Do NOT invent exotic diseases without evidence in the note and graph.
- If a common diagnosis clearly explains most findings, you may assign it a high probability.
- If the information is very sparse or contradictory, you may assign lower probabilities
  and express uncertainty in the justification.

Do not include any text outside the JSON object.
"""


PROMPT_MEDICO_CONSERVADOR = PROMPT_MEDICO_BASE + """
Behavioural profile:
- You are CONSERVATIVE.
- You strongly prefer common, well-supported diagnoses.
- You avoid rare conditions unless there is clear, strong evidence in both
  the clinical note and the knowledge graph.
- When in doubt, you prioritise safety and established guidelines, and you
  focus on the simplest diagnoses that explain the findings.
"""


PROMPT_MEDICO_EXPLORADOR = PROMPT_MEDICO_BASE + """
Behavioural profile:
- You are EXPLORATORY.
- In addition to common diagnoses, you also consider less frequent but plausible
  conditions that fit the combination of symptoms, risk factors and intermediate
  hypotheses in the graph.
- You are willing to assign small probabilities to rare but possible diagnoses
  if they are clinically coherent, while still keeping common diagnoses on top.
"""


# -------------------------------------------------------------------
# ESTRUTURA DE RESULTADO
# -------------------------------------------------------------------

@dataclass
class ResultadoMedico:
    medico_id: str
    diagnoses: List[Dict[str, Any]]


# -------------------------------------------------------------------
# CLASSE DO MÉDICO LLM
# -------------------------------------------------------------------

class MedicoLLM:
    """
    Wrapper para um 'médico virtual' baseado em LLM.
    """

    def __init__(
        self,
        medico_id: str,
        prompt_sistema: str,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
    ):
        self.medico_id = medico_id
        self.model = ChatOpenAI(model=model_name, temperature=temperature)

        # Prompt com placeholders para a nota e o grafo em JSON
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", prompt_sistema),
                (
                    "user",
                    "Clinical note:\n{nota}\n\n"
                    "Knowledge graph (JSON):\n{grafo_json}\n\n"
                    "Return ONLY the JSON object with the 'diagnoses' list."
                ),
            ]
        )

    def diagnosticar(self, nota: str, grafo_json: Dict[str, Any]) -> ResultadoMedico:
        """
        Envia a nota clínica + grafo para o LLM e devolve um ResultadoMedico
        com a lista de diagnósticos (cada um com name, probability, justification).
        """
        grafo_str = json.dumps(grafo_json, ensure_ascii=False)

        chain = self.prompt | self.model
        response = chain.invoke({"nota": nota, "grafo_json": grafo_str})

        raw = response.content

        # Parsing robusto do JSON de saída
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            text = str(raw).strip()
            inicio = text.find("{")
            fim = text.rfind("}") + 1
            if inicio == -1 or fim <= inicio:
                data = {}
            else:
                json_text = text[inicio:fim]
                try:
                    data = json.loads(json_text)
                except json.JSONDecodeError:
                    data = {}

        diagnoses = data.get("diagnoses", [])
        if not isinstance(diagnoses, list):
            diagnoses = []

        return ResultadoMedico(medico_id=self.medico_id, diagnoses=diagnoses)
