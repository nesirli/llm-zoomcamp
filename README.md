# LLM Zoomcamp 2026 — Nasir Nesirli

My coursework, notes, and project for [LLM Zoomcamp](https://github.com/DataTalks-Club/llm-zoomcamp) — a free 10-week course by [DataTalks.Club](https://datatalks.club) on building production-ready LLM applications.

## About the Course

LLM Zoomcamp teaches the engineering side of large language models: how to build retrieval-augmented generation (RAG) pipelines, implement vector and hybrid search, evaluate model outputs, and monitor live systems. The focus is on practical methods that make LLM-powered applications predictable and maintainable.

**Cohort:** June 2026 (live)  
**Format:** Pre-recorded lectures + weekly homework + final project  
**Certificate:** Awarded after completing the final project and peer-reviewing 3 others

## Curriculum

| Module | Topic | Status |
|--------|-------|--------|
| 01 | [Agentic RAG](./01-agentic-rag/) | ✅ |
| 02 | [Vector Search](./02-vector-search/) | 🔜 |
| 03 | [Orchestration](./03-orchestration/) | 🔜 |
| 04 | [Evaluation](./04-evaluation/) | 🔜 |
| 05 | [Monitoring](./05-monitoring/) | 🔜 |
| 06 | [Best Practices](./06-best-practices/) | 🔜 |
| 07 | [Projects](./07-projects/) | 🔜 |

## Repository Structure

```
llm-zoomcamp/
├── 01-agentic-rag/         # RAG fundamentals, agentic patterns
├── 02-vector-search/       # Embeddings, semantic search, PGVector
├── 03-orchestration/       # AI orchestration and pipelines
├── 04-evaluation/          # LLM-as-a-Judge, retrieval metrics
├── 05-monitoring/          # Grafana dashboards, feedback loops
├── 06-best-practices/      # Hybrid search, reranking
└── 07-projects/            # My end-to-end RAG projects
```

> Folders will be added as modules are completed.

## Tech Stack

- **LLM APIs:** OpenAI
- **Search:** minsearch, sqlitesearch, PGVector
- **Embeddings:** sentence-transformers
- **Agents:** OpenAI Responses API, ToyAIKit
- **Monitoring:** Grafana
- **Interface:** Streamlit / FastAPI

## Projects

Personal RAG projects live in [`07-projects/`](./07-projects/). Each will include a searchable knowledge base, a full RAG pipeline, evaluation, a user-facing interface, and a monitoring layer.

## Official Resources

- [Course GitHub](https://github.com/DataTalks-Club/llm-zoomcamp)
- [Video Lectures (YouTube)](https://www.youtube.com/@DataTalksClub)
- [DataTalks.Club Community (Slack)](https://datatalks.club/slack.html)

## License

Code in this repository is provided for learning and portfolio purposes. Course materials belong to DataTalks.Club.
