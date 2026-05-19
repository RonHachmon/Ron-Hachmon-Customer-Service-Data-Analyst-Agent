# Ron-Hachmon-Customer-Service-Data-Analyst-Agent

A LangGraph ReAct agent that answers questions about the [Bitext Customer Service](https://huggingface.co/datasets/bitext/Bitext-customer-support-llm-chatbot-training-dataset) dataset. Currently at **Phase 0** — dependencies installed and Nebius Token Factory connectivity verified.

## Requirements

- Python **3.11+**
- A [Nebius Token Factory](https://docs.tokenfactory.nebius.com/) API key

## Setup

```powershell
# 1. Clone and enter the repo
git clone <repo-url>
cd Ron-Hachmon-Customer-Service-Data-Analyst-Agent

# 2. Create and activate a virtual env (Windows PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 3. Install the project (editable) and its dependencies
pip install -e .

# 4. Configure your Nebius API key
copy .env.template .env
# then edit .env and set NEBIUS_API_KEY=<your-key>
```

On macOS/Linux replace step 2 with `python3 -m venv .venv && source .venv/bin/activate` and step 4 with `cp .env.template .env`.

## What you can run today

### 1. Load the Bitext dataset

Pulls the dataset from Hugging Face (cached locally after the first run) and prints three sample rows in debug mode.

```powershell
python main.py
```

### 2. Nebius smoke test

Verifies your API key, that both configured models are reachable, and that the reasoner model returns valid OpenAI-style tool calls. Exits 0 on success.

```powershell
python scripts/smoke_test_nebius.py
```

Expected output:

```
[router] OK -> route='structured' reason='...'
[reasoner] OK -> tool_call=echo_number(n='42')

All Phase 0 checks passed.
```

## Models

Two models are used (both [Nebius Token Factory](https://docs.tokenfactory.nebius.com/)):

| Role | Default model | Override via env var |
|---|---|---|
| Router (query classification) | `meta-llama/Meta-Llama-3.1-8B-Instruct` | `NEBIUS_ROUTER_MODEL` |
| Reasoner (ReAct loop + summarization) | `meta-llama/Llama-3.3-70B-Instruct` | `NEBIUS_REASONER_MODEL` |

The split keeps routing cheap and fast while giving the ReAct loop a stronger tool-caller. Override either env var if you want to swap models without touching code.

## Troubleshooting

- **`NEBIUS_API_KEY is not set`** — make sure `.env` exists and contains a non-empty `NEBIUS_API_KEY=...` line. The smoke test calls `load_dotenv()` so the value is picked up automatically.
- **`The model ... does not exist`** — Nebius changes its model catalog over time. Run `python -c "from openai import OpenAI; import os; print('\n'.join(m.id for m in OpenAI(api_key=os.environ['NEBIUS_API_KEY'], base_url='https://api.tokenfactory.nebius.com/v1/').models.list().data))"` to list what's live, then set `NEBIUS_ROUTER_MODEL` / `NEBIUS_REASONER_MODEL` accordingly.

## Roadmap

The full agent is being built in phases. Sections below will be added as each phase lands:

- Phase 1 — pandas-backed data layer
- Phase 2 — six typed tools with Pydantic schemas
- Phase 3 — query router (structured / unstructured / out-of-scope)
- Phase 4 — ReAct loop with max-iteration fallback
- Phase 5 — CLI with reasoning trace
- Phase 6 — episodic memory (SQLite checkpointer)
- Phase 7 — per-user profile
- Phase 8 — FastMCP server
