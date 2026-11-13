# RAG Evaluation Lab

A small RAG system that answers questions about a PDF, plus an evaluation harness that scores the answers with [DeepEval](https://github.com/confident-ai/deepeval) and a Streamlit app to try it and compare runs. It runs on free tools: **Groq** for the LLM, **local embeddings** for retrieval.

> Adapted from [Mohamad-Hachem/LLM_Evaluation_System](https://github.com/Mohamad-Hachem/LLM_Evaluation_System), which used OpenAI. See [What I changed](#what-i-changed).

## Features

- **Ask the PDF**: chat page that answers from `pdfs/Penguins_ACL.pdf` and shows the retrieved chunks behind every answer.
- **Evaluation results**: dashboard comparing saved runs on four metrics.
- **Eval pipeline**: 20 question/answer pairs scored for answer relevancy, faithfulness, context relevancy and correctness.

## How it works

```
PDF -> PyMuPDF paragraphs -> bge-small embeddings (local) -> in-memory vector store
question -> LangChain agent (Groq) -> retrieval tool -> answer + retrieved chunks
eval_dataset.jsonl -> runner -> DeepEval metrics (Groq judge) -> scored JSONL -> dashboard
```

| Part | Choice |
|---|---|
| LLM and judge | Groq, `openai/gpt-oss-120b` (override with `GROQ_MODEL`, `GROQ_JUDGE_MODEL`) |
| Embeddings | `BAAI/bge-small-en-v1.5` via fastembed, runs locally |
| Vector store | LangChain `InMemoryVectorStore` |
| Evaluation | DeepEval: `AnswerRelevancyMetric`, `FaithfulnessMetric`, `ContextualRelevancyMetric`, `GEval` (correctness) |
| UI | Streamlit |

## Run it

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/). Get a free key at [console.groq.com](https://console.groq.com).

```bash
uv sync
echo 'GROQ_API_KEY="your_key"' > .env
uv run streamlit run app.py
```

Run the evaluation from the repo root (`EVAL_LIMIT` caps the number of questions, useful on Groq's free tier):

```bash
EVAL_LIMIT=5 PYTHONPATH=. uv run python versions/v3/eval_runner3.py
PYTHONPATH=. uv run python versions/v3/eval_all_metrics3_fast.py
```

Single-metric examples live in `single_tests/`.

## Evaluation results

`experiment_v1/v2/v3.jsonl` are the saved summaries shown on the dashboard. **They were produced by the original project with OpenAI models**, not Groq. Scores from an LLM judge vary between models and runs, so numbers from the Groq setup will differ. Rerun the scripts above to produce your own.

| Run | Answer relevancy | Faithfulness | Context relevancy | Correctness |
|---|---:|---:|---:|---:|
| experiment_v1 | 0.988 | 1.000 | 0.733 | 0.900 |
| experiment_v2 | 1.000 | 1.000 | 0.781 | 0.915 |
| experiment_v3 | 1.000 | 1.000 | 0.787 | 0.890 |

## What I changed

- Replaced OpenAI (`gpt-4o`, `text-embedding-3-large`, the default DeepEval judge) with Groq plus local embeddings, so the project needs no paid API.
- Added a custom DeepEval judge (`RAG1/judge.py`) backed by Groq.
- The agent now returns the chunks the retriever actually returned instead of context the LLM reports, and no longer relies on structured-output tool calls, which were unreliable on Groq.
- Evaluation runs one request at a time and supports `EVAL_LIMIT`, to fit Groq's free-tier token limits.
- Added the Streamlit chat page and multipage app (`app.py`, `chat.py`).
- Fixed the tool name in the system prompt (it said `collection_info`, the tool is `collecting_info`).

## Limitations

- The app is tied to one PDF (hardcoded in `RAG1/indexing.py`).
- Free-tier Groq limits can interrupt long evaluation runs.
