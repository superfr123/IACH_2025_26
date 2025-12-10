# grafo_conhecimento.py

from dataclasses import dataclass
from typing import Dict, Any, List
import json
import os
from collections import deque

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pyvis.network import Network


PROMPT_GRAFO = """
You are a medical knowledge extraction and reasoning assistant.

You receive a raw clinical note about a patient, which may include:
- demographics (age, sex, etc.)
- habits and risk factors (smoking, alcohol, obesity, etc.)
- past medical history and comorbidities
- current symptoms and signs
- treatments and tests mentioned

Your task is to build a structured knowledge graph that makes explicit
how symptoms and risk factors combine into intermediate hypotheses and
lead to one or more probable diagnoses.

The graph should have multiple "layers", not just everything pointing to
the patient. Use the following conventions:

NODES
- Return a JSON object with a field "nodes": a list of nodes.
- Each node must have:
    - "id": unique string identifier (e.g., "n1", "n2", ...)
    - "label": short human-readable name
    - "type": one of:
        - "patient"
        - "demographic"
        - "symptom"
        - "sign"
        - "risk_factor"
        - "habit"
        - "comorbidity"
        - "test"
        - "treatment"
        - "intermediate_hypothesis"   (e.g., "respiratory infection", "heart failure syndrome")
        - "diagnosis"                 (e.g., "pneumonia", "congestive heart failure")
        - "other"

EDGES
- Return also a field "edges": a list of edges.
- Each edge must have:
    - "source": id of source node
    - "target": id of target node
    - "relation": short verb phrase, for example:
        - "has_symptom", "has_sign"
        - "has_risk_factor", "has_comorbidity"
        - "received_test", "received_treatment"
        - "supports", "compatible_with", "part_of", "worsens"
        - "suggests_hypothesis"       (symptom/risk_factor -> intermediate_hypothesis)
        - "supports_diagnosis"        (intermediate_hypothesis -> diagnosis)
        - "suggests_diagnosis"        (symptom/sign directly -> diagnosis when appropriate)

STRUCTURE
- Include one central "patient" node.
- Link demographics, risk factors, comorbidities, tests and treatments directly
  to the patient when appropriate.
- Link symptoms and signs to one or more "intermediate_hypothesis" nodes when
  they naturally cluster (e.g., "respiratory syndrome", "cardiac decompensation").
- Link "intermediate_hypothesis" nodes to one or more "diagnosis" nodes using
  "supports_diagnosis".
- You may also link symptoms directly to diagnoses with "suggests_diagnosis"
  when there is a strong direct pattern.

CONNECTIVITY (IMPORTANT)
- Ideally, every "diagnosis" node should be clinically related to the patient.
- However, if some parts of the text describe other possibilities or
  background information that you cannot clearly connect to the current
  patient, you may leave those nodes disconnected.
- Do NOT fabricate connections that are not supported by the note.

OUTPUT
- Return ONLY a JSON object with exactly two top-level keys:
  "nodes" and "edges".
- Do not include explanations, comments or markdown.
"""


@dataclass
class GrafoResultado:
    grafo_json: Dict[str, Any]
    html_path: str
    num_componentes: int = 1  # para análise posterior


class ConstrutorGrafoLLM:
    def __init__(self, model_name: str = "gpt-4o-mini"):
        self.model = ChatOpenAI(model=model_name, temperature=0)
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", PROMPT_GRAFO),
                ("user", "Clinical note:\n{nota}\n\nReturn ONLY the JSON object."),
            ]
        )

    def construir(self, nota: str, output_dir: str, nome_base: str = "grafo") -> GrafoResultado:
        os.makedirs(output_dir, exist_ok=True)

        chain = self.prompt | self.model
        response = chain.invoke({"nota": nota})
        raw_text = response.content

        grafo_json = self._parse_json(raw_text)

        # só analisamos quantas componentes há, não mexemos nas arestas
        num_comp = self._count_components(grafo_json)

        html_path = self._criar_html(grafo_json, output_dir, nome_base)

        return GrafoResultado(grafo_json=grafo_json, html_path=html_path, num_componentes=num_comp)

    # ---------------------- helpers internos -------------------------

    def _parse_json(self, raw_text: str) -> Dict[str, Any]:
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            inicio = raw_text.find("{")
            fim = raw_text.rfind("}") + 1
            json_text = raw_text[inicio:fim]
            return json.loads(json_text)

    def _count_components(self, grafo_json: Dict[str, Any]) -> int:
        nodes: List[Dict[str, Any]] = grafo_json.get("nodes", []) or []
        edges: List[Dict[str, Any]] = grafo_json.get("edges", []) or []

        node_ids = [n.get("id") for n in nodes if n.get("id") is not None]
        if not node_ids:
            return 0

        adj = {nid: set() for nid in node_ids}
        for e in edges:
            s = e.get("source")
            t = e.get("target")
            if s in adj and t in adj:
                adj[s].add(t)
                adj[t].add(s)

        visited = set()
        comp_count = 0

        for nid in node_ids:
            if nid in visited:
                continue
            comp_count += 1
            q = deque([nid])
            visited.add(nid)
            while q:
                u = q.popleft()
                for v in adj.get(u, []):
                    if v not in visited:
                        visited.add(v)
                        q.append(v)

        return comp_count

    def _criar_html(self, grafo_json: Dict[str, Any], output_dir: str, nome_base: str) -> str:
        net = Network(height="700px", width="100%", directed=True)

        for node in grafo_json.get("nodes", []):
            node_id = node.get("id")
            label = node.get("label", node_id)
            tipo = (node.get("type") or "").lower()
            color = self._cor_por_tipo(tipo)
            net.add_node(node_id, label=label, color=color, title=tipo)

        for edge in grafo_json.get("edges", []):
            source = edge.get("source")
            target = edge.get("target")
            relation = edge.get("relation", "")
            net.add_edge(source, target, label=relation)

        filename = f"{nome_base}.html"
        path = os.path.join(output_dir, filename)
        net.write_html(path)
        return path

    def _cor_por_tipo(self, tipo: str) -> str:
        if tipo == "patient":
            return "green"
        if "symptom" in tipo or "sign" in tipo:
            return "blue"
        if "risk_factor" in tipo or "habit" in tipo:
            return "orange"
        if "comorbidity" in tipo:
            return "purple"
        if "intermediate" in tipo or "hypothesis" in tipo:
            return "gold"
        if "diagnosis" in tipo:
            return "red"
        return "gray"
