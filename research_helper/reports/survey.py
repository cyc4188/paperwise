"""Mode 1: Generate a domain survey report from multiple papers."""
from __future__ import annotations
import json
from pathlib import Path

from research_helper import config
from research_helper.llm import client as llm
from research_helper.readers.arxiv_reader import PaperMeta

SYSTEM_PROMPT = """\
You are an expert research assistant who writes comprehensive academic survey reports in Chinese.\
"""

SURVEY_PROMPT = """\
以下是关于"{query}"领域的 {n} 篇论文的摘要信息，请生成一份完整的领域综述报告，包含以下结构：

1. 领域概述：该领域的核心问题和研究意义
2. 主要研究方向：归纳出 3-5 个子方向，每个子方向列出代表论文
3. 发展脉络：按时间梳理该领域的重要进展
4. 方法对比：不同方法的优缺点比较
5. 开放问题与未来趋势：当前局限和值得探索的方向
6. 参考文献列表：所有论文的完整引用

---

论文列表（格式：标题 | 作者 | 时间 | Arxiv ID | 摘要）：

{papers_block}
"""


def _build_papers_block(papers: list[PaperMeta], max_abstract: int) -> str:
    lines = []
    for p in papers:
        authors = ", ".join(p.authors[:3]) + (" et al." if len(p.authors) > 3 else "")
        abstract = p.abstract[:max_abstract]
        lines.append(f"**{p.title}** | {authors} | {p.published} | {p.arxiv_id}\n{abstract}")
    return "\n\n".join(lines)


def generate(query: str, papers: list[PaperMeta], output_dir: Path) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "survey.md"

    papers_block = _build_papers_block(papers, config.SURVEY_MAX_ABSTRACT_CHARS)
    prompt = SURVEY_PROMPT.format(query=query, n=len(papers), papers_block=papers_block)

    report_md = llm.complete(SYSTEM_PROMPT, prompt, max_tokens=10000)

    # Prepend metadata header
    header = (
        f"# 领域综述：{query}\n\n"
        f"**检索论文数**：{len(papers)}\n"
        f"**报告生成时间**：{_today()}\n\n"
        f"---\n\n"
    )
    report_path.write_text(header + report_md, encoding="utf-8")

    # Save paper list as JSON for reference
    meta_path = output_dir / "papers.json"
    meta_path.write_text(
        json.dumps(
            [{"arxiv_id": p.arxiv_id, "title": p.title, "published": p.published} for p in papers],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return report_path


def _today() -> str:
    from datetime import date
    return date.today().isoformat()
