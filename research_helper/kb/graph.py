"""Build a knowledge graph from all papers in the KB / outputs directory."""
from __future__ import annotations
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from research_helper import config
from research_helper.llm import client as llm

# ── Prompts ──────────────────────────────────────────────────────────────────

_SYSTEM = "你是论文信息提取助手。只输出合法 JSON，不加 markdown 代码块或任何其他内容。"

_EXTRACT_PROMPT = (
    "从以下论文精读报告中提取结构化信息。\n\n"
    "输出严格 JSON，包含两个字段：\n"
    "1. concepts: 列表，3-8 个核心概念/方法标签，每个不超过 5 词，优先英文专有名词。"
    '例如 ["KV Cache", "RoPE", "RAG", "Speculative Decoding"]\n'
    "2. relations: 列表，该论文与其他论文的显式关系。每项包含："
    " target_title（被比较论文标题）, target_arxiv（arxiv ID 或 null）,"
    ' relation（builds_on / compares_to / extends / contradicts）, description（一句话中文说明）。\n\n'
    "报告（前 6000 字）：\n{report}"
)

# ── Data classes ──────────────────────────────────────────────────────────────

EdgeType = Literal["similar_to", "builds_on", "compares_to", "extends", "contradicts", "uses"]


@dataclass
class GNode:
    id: str
    type: Literal["paper", "concept"]
    label: str
    arxiv_id: str = ""
    published: str = ""
    categories: list[str] = field(default_factory=list)


@dataclass
class GEdge:
    source: str
    target: str
    type: EdgeType
    weight: float = 1.0
    description: str = ""


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_info(paper_dir: Path, report: str) -> dict:
    cache = paper_dir / "graph_info.json"
    if cache.exists():
        return json.loads(cache.read_text(encoding="utf-8"))

    raw = llm.complete(_SYSTEM, _EXTRACT_PROMPT.format(report=report[:6000]), max_tokens=3000)
    raw = raw.strip()
    # Strip markdown code fences
    if raw.startswith("```"):
        lines = raw.splitlines()
        lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        raw = "\n".join(lines).strip()

    try:
        info = json.loads(raw)
    except Exception:
        # Last-resort: extract first {...} block
        import re as _re
        m = _re.search(r"\{.*\}", raw, _re.DOTALL)
        try:
            info = json.loads(m.group()) if m else {}
        except Exception:
            info = {}

    info.setdefault("concepts", [])
    info.setdefault("relations", [])

    cache.write_text(json.dumps(info, ensure_ascii=False, indent=2), encoding="utf-8")
    return info


# ── Graph builder ─────────────────────────────────────────────────────────────

def build(
    outputs_dir: Path | None = None,
    similarity_threshold: float = 0.55,
    progress_cb=None,
) -> tuple[list[GNode], list[GEdge]]:
    """Build graph from all papers in outputs_dir. Returns (nodes, edges)."""
    if outputs_dir is None:
        outputs_dir = config.OUTPUTS_DIR

    paper_dirs = sorted(
        d for d in outputs_dir.iterdir()
        if d.is_dir() and (d / "meta.json").exists()
    )

    nodes: list[GNode] = []
    edges: list[GEdge] = []
    paper_nodes: dict[str, GNode] = {}       # arxiv_id → GNode
    concept_nodes: dict[str, GNode] = {}     # label → GNode
    title_to_arxiv: dict[str, str] = {}      # lower title[:60] → arxiv_id

    # ── Pass 1: paper nodes ───────────────────────────────────────────────────
    paper_infos: list[tuple[str, dict]] = []

    for i, paper_dir in enumerate(paper_dirs):
        if progress_cb:
            progress_cb(f"Processing {paper_dir.name} ({i+1}/{len(paper_dirs)})…")

        meta = json.loads((paper_dir / "meta.json").read_text(encoding="utf-8"))
        arxiv_id = meta["arxiv_id"]
        title = meta["title"]

        node = GNode(
            id=f"paper:{arxiv_id}",
            type="paper",
            label=title,
            arxiv_id=arxiv_id,
            published=meta.get("published", ""),
            categories=meta.get("categories", []),
        )
        nodes.append(node)
        paper_nodes[arxiv_id] = node
        title_to_arxiv[title.lower()[:60]] = arxiv_id

        report_path = paper_dir / "report.md"
        if report_path.exists():
            info = _extract_info(paper_dir, report_path.read_text(encoding="utf-8"))
            paper_infos.append((arxiv_id, info))

    # ── Pass 2: concept nodes + paper→concept edges ───────────────────────────
    for arxiv_id, info in paper_infos:
        for concept in info.get("concepts", []):
            concept = concept.strip()
            if not concept:
                continue
            if concept not in concept_nodes:
                cnode = GNode(id=f"concept:{concept}", type="concept", label=concept)
                concept_nodes[concept] = cnode
                nodes.append(cnode)
            edges.append(GEdge(
                source=f"paper:{arxiv_id}",
                target=f"concept:{concept}",
                type="uses",
            ))

    # ── Pass 3: explicit relation edges ──────────────────────────────────────
    for arxiv_id, info in paper_infos:
        for rel in info.get("relations", []):
            target = rel.get("target_arxiv") or title_to_arxiv.get(
                rel.get("target_title", "").lower()[:60]
            )
            if not target or target not in paper_nodes:
                continue
            edge_type = rel.get("relation", "compares_to")
            if edge_type not in ("builds_on", "compares_to", "extends", "contradicts"):
                edge_type = "compares_to"
            edges.append(GEdge(
                source=f"paper:{arxiv_id}",
                target=f"paper:{target}",
                type=edge_type,
                description=rel.get("description", ""),
            ))

    # ── Pass 4: similarity edges from vector store ───────────────────────────
    if len(paper_nodes) >= 2:
        if progress_cb:
            progress_cb("Computing similarity edges…")
        _add_similarity_edges(paper_nodes, edges, similarity_threshold)

    return nodes, edges


def _add_similarity_edges(
    paper_nodes: dict[str, GNode],
    edges: list[GEdge],
    threshold: float,
) -> None:
    try:
        from research_helper.kb import store
    except Exception:
        return

    existing_sim: set[tuple[str, str]] = set()

    for arxiv_id, node in paper_nodes.items():
        try:
            results = store.query(node.label, top_k=len(paper_nodes) * 3)
        except Exception:
            continue
        for r in results:
            if r.arxiv_id == arxiv_id or r.arxiv_id not in paper_nodes:
                continue
            sim = 1 - r.distance
            if sim < threshold:
                continue
            key = tuple(sorted([arxiv_id, r.arxiv_id]))
            if key in existing_sim:
                continue
            existing_sim.add(key)
            a, b = sorted([f"paper:{arxiv_id}", f"paper:{r.arxiv_id}"])
            edges.append(GEdge(source=a, target=b, type="similar_to", weight=round(sim, 3)))


# ── Export ────────────────────────────────────────────────────────────────────

def export_json(nodes: list[GNode], edges: list[GEdge], out_path: Path) -> None:
    data = {
        "nodes": [vars(n) for n in nodes],
        "edges": [vars(e) for e in edges],
    }
    out_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


_EDGE_COLORS: dict[str, str] = {
    "similar_to":  "#4a9eff",
    "builds_on":   "#00d4aa",
    "extends":     "#00d4aa",
    "compares_to": "#ff9944",
    "contradicts": "#ff4444",
    "uses":        "#666666",
}

_LEGEND_HTML = """
<div style="position:fixed;bottom:20px;left:20px;background:#1a1a2e;
            border:1px solid #444;border-radius:8px;padding:14px;font-family:sans-serif;z-index:999">
  <b style="color:#fff;font-size:13px">图例</b><br><br>
  <span style="color:#4a9eff">●</span>
  <span style="color:#eee;font-size:12px"> 论文节点</span><br>
  <span style="color:#ff9944">◆</span>
  <span style="color:#eee;font-size:12px"> 概念节点</span><br><br>
  <span style="color:#4a9eff;font-size:11px">━━</span>
  <span style="color:#eee;font-size:11px"> similar_to (向量相似)</span><br>
  <span style="color:#00d4aa;font-size:11px">━━</span>
  <span style="color:#eee;font-size:11px"> builds_on / extends</span><br>
  <span style="color:#ff9944;font-size:11px">━━</span>
  <span style="color:#eee;font-size:11px"> compares_to</span><br>
  <span style="color:#ff4444;font-size:11px">━━</span>
  <span style="color:#eee;font-size:11px"> contradicts</span><br>
  <span style="color:#666;font-size:11px">━━</span>
  <span style="color:#eee;font-size:11px"> uses (概念)</span>
</div>
"""


def export_html(nodes: list[GNode], edges: list[GEdge], out_path: Path) -> None:
    from pyvis.network import Network

    net = Network(
        height="100vh",
        width="100%",
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        notebook=False,
    )
    net.force_atlas_2based(gravity=-60, central_gravity=0.005, spring_length=180, damping=0.9)

    for node in nodes:
        if node.type == "paper":
            tooltip = f"<b>{node.label}</b><br>{node.arxiv_id}<br>{node.published}"
            if node.categories:
                tooltip += f"<br>{'、'.join(node.categories)}"
            net.add_node(
                node.id,
                label=node.label[:40],
                title=tooltip,
                shape="dot",
                size=22,
                color={"background": "#4a9eff", "border": "#2277cc",
                       "highlight": {"background": "#77bbff", "border": "#4a9eff"}},
                font={"size": 13, "color": "white", "strokeWidth": 2, "strokeColor": "#111"},
            )
        else:
            net.add_node(
                node.id,
                label=node.label,
                title=f"概念: {node.label}",
                shape="diamond",
                size=14,
                color={"background": "#ff9944", "border": "#cc6611",
                       "highlight": {"background": "#ffbb77", "border": "#ff9944"}},
                font={"size": 11, "color": "#dddddd", "strokeWidth": 1, "strokeColor": "#111"},
            )

    for edge in edges:
        color = _EDGE_COLORS.get(edge.type, "#888888")
        is_sim = edge.type == "similar_to"
        width = max(1.0, edge.weight * 4) if is_sim else 2.0
        tooltip = edge.description or edge.type
        if is_sim:
            tooltip = f"相似度: {edge.weight:.2f}"
        net.add_edge(
            edge.source,
            edge.target,
            title=tooltip,
            color={"color": color, "highlight": color, "opacity": 0.8},
            width=width,
            dashes=is_sim,
            arrows="" if is_sim else "to",
        )

    net.set_options(json.dumps({
        "interaction": {
            "hover": True,
            "navigationButtons": True,
            "tooltipDelay": 100,
        },
        "physics": {
            "enabled": True,
            "stabilization": {"iterations": 300, "updateInterval": 25},
        },
    }))

    html = net.generate_html()
    # Inject legend before </body>
    html = html.replace("</body>", _LEGEND_HTML + "\n</body>")
    out_path.write_text(html, encoding="utf-8")
