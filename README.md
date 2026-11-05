# 🛍️ ShopAssist AI

An AI-powered customer support agent built with **LangGraph, RAG, Groq, FAISS, Sentence Transformers, PostgreSQL, and Streamlit**.

ShopAssist AI understands customer requests, classifies intent, retrieves company policies, interacts with an order database, maintains conversation context, and escalates unsupported requests to human support.

**It is not just an LLM chatbot — it is a stateful agentic workflow where the LLM is one component inside a larger system.**

---

## 📑 Table of Contents

- [Live Demo](#-live-demo)
- [Features](#-features)
- [Architecture](#️-architecture)
- [How It Works](#-how-it-works)
- [Safety & Design Decisions](#-safety--design-decisions)
- [Tech Stack](#️-tech-stack)
- [Project Structure](#-project-structure)
- [Setup](#️-setup)
- [Conversation Persistence](#-conversation-persistence)
- [Database Schema](#-database-schema)
- [Testing](#-testing)
- [Development Scripts](#️-development-scripts)
- [Evaluation](#-evaluation)
- [Example Requests](#-example-requests)
- [Deployment](#-deployment)
- [Known Limitations & Future Work](#️-known-limitations--future-work)
- [Troubleshooting](#-troubleshooting)
- [Security Notes](#-security-notes)
- [Author](#-author)
- [License](#-license)

---

## 🚀 Live Demo

> 🌐 **Live App:** _Coming soon — deploying to Streamlit Community Cloud_

---

## ✨ Features

- 🤖 AI-powered intent classification using an LLM
- 📚 Retrieval-Augmented Generation (RAG) for policy questions
- 🔎 Semantic search using FAISS + Sentence Transformers
- 🗄️ PostgreSQL order database (tracking + cancellation)
- 💬 Persistent conversation memory via LangGraph + PostgreSQL checkpointer
- 🔀 Conditional workflow routing with LangGraph
- 🛑 Confirmation flow for destructive actions (order cancellation)
- 👤 Human-support escalation for unsupported or low-confidence requests
- 🚦 RAG relevance threshold to prevent hallucination
- 🧯 Retry logic with exponential backoff for LLM calls
- 🌐 Professional Streamlit UI with quick-action chips and sidebar
- 📊 28-case evaluation suite with **100% pass rate**

---

## 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │   Streamlit UI      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   LangGraph Agent   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Analyze Request     │
                    │ (Groq LLM)          │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Route Request       │
                    └──┬─────┬─────┬──────┘
                       │     │     │
              ┌────────┘     │     └────────┐
              ▼              ▼              ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │   RAG    │   │  Tools   │   │Escalation│
        │  FAISS   │   │PostgreSQL│   │  Human   │
        └────┬─────┘   └────┬─────┘   └────┬─────┘
             │              │              │
             └──────┬───────┘              │
                    ▼                      │
             ┌──────────────┐              │
             │   Generate   │              │
             │   Response   │              │
             └──────┬───────┘              │
                    │                      │
                    └──────────┬───────────┘
                               ▼
                          Customer
```

---

## 🧠 How It Works

When a customer sends a message, the LangGraph workflow runs:

1. **Analyze** — Groq LLM classifies intent + extracts order ID (JSON schema enforced)
2. **Route** — LangGraph decides: RAG / Tool / Escalation / Conversational
3. **Execute** — Retrieve policy docs, call an order tool, or handle conversation
4. **Generate** — Groq LLM produces a grounded answer using only retrieved context
5. **Persist** — Every state change is checkpointed to PostgreSQL

### Supported Intents

| Category | Intents |
|---|---|
| Policy (RAG) | `return_policy`, `refund_policy`, `shipping_policy`, `policy_unknown` |
| Order (Tool) | `order_tracking`, `cancel_order` |
| Escalation | `human_handoff`, `complaint`, `agent_request` |
| Conversational | `greeting`, `thanks`, `capabilities`, `out_of_scope` |

---

## 🔐 Safety & Design Decisions

These are the decisions that separate this project from a naive "LLM wrapper":

- **LLM never mutates the database directly.** The LLM classifies intent; Python decides what action is allowed (business rules live in code, not in prompts).
- **Cancellation requires confirmation.** The agent asks "Are you sure?" and only executes on explicit user confirmation.
- **RAG relevance threshold.** Low-similarity matches are filtered out, so the agent says "I don't know" instead of hallucinating policies that don't exist.
- **Short-circuit for common conversational inputs.** "thanks" and "hi" bypass the LLM entirely, avoiding context-leakage misclassifications.
- **Stale state is cleared every turn.** `retrieved_docs`, `tool_result`, and `response` are reset at the start of each request.
- **Retries on LLM calls.** Transient Groq failures are retried with exponential backoff using `tenacity`.

---

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python (3.11+) | Core language |
| LangGraph | Stateful agent workflow + checkpointing |
| LangChain | LLM / RAG integrations |
| Groq | LLM inference API |
| GPT-OSS-20B | Language model |
| FAISS | Vector similarity search |
| Sentence Transformers | Local embeddings (`all-MiniLM-L6-v2`) |
| PostgreSQL (Supabase) | Order data + LangGraph checkpoints |
| SQLAlchemy | ORM |
| Pydantic | Structured output validation |
| Streamlit | Web UI |
| tenacity | Retry logic |

---

## 📁 Project Structure

```text
shopassist-ai/
├── app.py                     # Streamlit UI
├── requirements.txt
├── .env.example
├── README.md
├── knowledge_base/            # RAG source documents
│   ├── return_policy.md
│   ├── refund_policy.md
│   └── shipping_policy.md
├── vectorstore/               # Generated by ingestion — MUST be committed for deployment
│   ├── index.faiss
│   └── index.pkl
├── src/
│   ├── graph.py               # LangGraph workflow
│   ├── state.py               # Graph state schema
│   ├── schemas.py             # Pydantic models
│   ├── database/
│   │   ├── connection.py
│   │   └── models.py
│   ├── rag/
│   │   ├── ingest.py
│   │   └── retriever.py
│   └── tools/
│       └── order_tools.py
└── tests/
    ├── evaluation_cases.py    # 28 evaluation test cases
    ├── evaluate.py            # Evaluation runner (100% pass rate)
    ├── test_database.py       # DB connectivity test
    ├── test_order_tools.py    # Order tool unit tests
    ├── test_retriever.py      # RAG retriever test
    ├── test_graph.py          # Full workflow test with assertions
    │
    ├── create_tables.py       # Creates DB tables from SQLAlchemy models
    ├── seed_demo_orders.py    # Seeds demo orders 2001–2300 (main script)
    ├── seed_orders.py         # Alternative seeding script (dev only)
    └── add_order_1027.py      # Utility: insert a single specific order (dev only)
```

---

## ⚙️ Setup

### Prerequisites

- **Python 3.11 or higher**
- **Groq API key** — free at [console.groq.com](https://console.groq.com/keys)
- **PostgreSQL database** — free tier on [Supabase](https://supabase.com) or [Neon](https://neon.tech)

### 1. Clone and install

```bash
git clone <YOUR_REPO_URL>
cd shopassist-ai
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
```

Fill in your `.env`:

```env
GROQ_API_KEY=your_groq_api_key
DATABASE_URL=postgresql://user:password@host:5432/dbname
```

> **Important:** The `DATABASE_URL` must **not** contain `+psycopg`. It should be a plain `postgresql://...` connection string.

### 3. Set up the database

Create the `orders` table and LangGraph checkpointer tables:

```bash
python -m tests.create_tables
```

Seed demo orders (range **2001–2300** with mixed statuses — `processing`, `shipped`, `cancelled` — so all tool paths are testable):

```bash
python -m tests.seed_demo_orders
```

### 4. Build the vector store

```bash
python -m src.rag.ingest
```

> **Important:** Re-run this whenever you edit files in `knowledge_base/`. The FAISS index is a build artifact and becomes stale the moment a source document changes.

### 5. Run

```bash
streamlit run app.py
```

The app opens at `http://localhost:8501`.

---

## 💬 Conversation Persistence

ShopAssist AI remembers conversation history across turns **and across process restarts**, using LangGraph's PostgreSQL checkpointer.

**How it works:**

- Every Streamlit session gets a unique `thread_id` (a UUID generated in `app.py`).
- LangGraph saves a **checkpoint** of the full graph state to PostgreSQL after every node execution.
- On the next turn, LangGraph loads the latest checkpoint for that `thread_id` and resumes with the full state intact.
- This is how the agent remembers an order ID across follow-ups like "Where is order 2002?" → "Give me its tracking number."

**Where it is stored:**

Checkpoints live in the same Supabase/PostgreSQL database, but in **separate tables** from your business data:

| Table | Purpose |
|---|---|
| `orders` | Business data — order information |
| `checkpoints` | LangGraph conversation state (per thread) |
| `checkpoint_blobs` | Serialized payloads for checkpoints |
| `checkpoint_writes` | Intermediate writes during node execution |

You can inspect them from the Supabase SQL Editor:

```sql
SELECT thread_id, checkpoint_id, created_at
FROM checkpoints
ORDER BY created_at DESC
LIMIT 10;
```

**Resetting a conversation** — Click "🔄 Start new conversation" in the sidebar. This generates a fresh `thread_id`, so the agent treats it as a new session.

---

## 🗄️ Database Schema

### `orders` table

| Column | Type | Notes |
|---|---|---|
| `order_id` | `String(50)` | Primary key |
| `customer_id` | `String(50)`, nullable | Indexed |
| `status` | `String(50)` | Indexed — see statuses below |
| `tracking_number` | `String(100)`, nullable | e.g. `TRK2002` |
| `estimated_delivery` | `Date`, nullable | ISO date |

### Order statuses

| Status | Meaning | Cancellable? |
|---|---|---|
| `processing` | Not yet shipped | ✅ Yes |
| `shipped` | Handed to carrier | ❌ No |
| `cancelled` | Cancelled by user | ❌ No (already cancelled) |

The business rule "only `processing` orders can be cancelled" lives in `src/tools/order_tools.py`, **not** in the LLM prompt. This is intentional — the LLM decides intent, Python enforces rules.

### Demo data

`tests/seed_demo_orders.py` seeds **300 orders** in the range **2001–2300** with a realistic mix of statuses, so every tool path (track, cancel-success, cancel-rejection, not-found) can be tested without a real e-commerce backend.

---

## 🧪 Testing

```bash
python -m tests.test_database
python -m tests.test_order_tools
python -m tests.test_retriever
python -m tests.test_graph       # full workflow test with assertions
```

---

## 🛠️ Development Scripts

The `tests/` folder contains utility scripts used during development. They are **not part of the main application workflow** and do not need to run in production.

| Script | Required? | Purpose |
|---|---|---|
| `create_tables.py` | ✅ Yes | Creates the `orders` and checkpointer tables. Run once on a fresh database. |
| `seed_demo_orders.py` | ✅ Yes | Seeds 300 demo orders (`2001–2300`) with mixed statuses. Required for the demo to work. |
| `seed_orders.py` | ❌ No | Alternative seeding script used during early development. Kept for reference. |
| `add_order_1027.py` | ❌ No | One-off utility to insert a single specific order (`1027`) for manual testing. |

**The production flow is:**

```
src/graph.py
    ↓
src/tools/order_tools.py
    ↓
src/database/
```

The scripts above are helpers, not part of the runtime.

---

## 📊 Evaluation

ShopAssist AI ships with a **28-case evaluation suite** that measures routing accuracy, response correctness, and safety behavior across 8 categories: RAG, order tracking, cancellation, escalation, conversational intents, context-leakage prevention, multi-turn memory, and multi-intent handling.

### How to run

```bash
python -m tests.evaluate
```

The runner sends each case through the full LangGraph workflow using a fresh `thread_id`, then asserts three things per case:

1. **Intent** — the classified intent matches expectation
2. **Route** — the chosen graph path is correct
3. **Response** — required keywords present, forbidden keywords absent

### Latest results

```text
======================================================================
SHOPASSIST AI — EVALUATION REPORT
======================================================================
CATEGORY                 PASSED / TOTAL
----------------------------------------------------------------------
cancellation               4 / 4
context_leakage            3 / 3
conversational             4 / 4
escalation                 3 / 3
memory                     2 / 2
multi_intent               1 / 1
order_tracking             4 / 4
rag                        7 / 7
----------------------------------------------------------------------
TOTAL                     28 / 28   (100.0%)
======================================================================
```

### Design notes

- The suite is **deterministic** — no LLM-as-judge, no external scoring service. Each test asserts concrete outcomes.
- Where the LLM can legitimately choose between **semantically equivalent intents** (e.g., `human_handoff` vs `agent_request` — both escalate), the test accepts any intent that produces the correct behavior. **We measure outcomes, not internal labels.**
- Apostrophes and dashes are normalized before keyword matching, because LLMs frequently emit typographic quotes (`’`) rather than ASCII (`'`).

### Known limitations of the suite

- Small test set (28 cases) — sufficient for a demo, not for production regression testing.
- No measurement of latency, token cost, or retrieval quality scores (precision/recall of retrieved documents).
- No adversarial cases (prompt injection, jailbreaks).

---

## 💡 Example Requests

| Category | Example |
|---|---|
| Policy | `What is your refund policy?` |
| Policy follow-up | `How many days for a UPI refund?` |
| Order tracking | `Where is my order 2002?` |
| Follow-up | `Give me its tracking number.` |
| Cancellation | `I want to cancel order 2015` → `yes` |
| Escalation | `I want to speak to a human agent` |
| Conversational | `hi`, `thanks`, `what can you do?` |

---

## 🚢 Deployment

This app is designed for **Streamlit Community Cloud** (free tier).

### 1. Push to GitHub

Make sure `.gitignore` **does not** ignore `vectorstore/`. Streamlit Cloud needs the FAISS index files to be committed.

Also make sure `.env` **is** ignored. Add these to `.gitignore`:

```
.env
.env.*
!.env.example
__pycache__/
*.pyc
.venv/
*.log
.streamlit/secrets.toml
```

### 2. Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"** → connect your GitHub repo
3. Set main file path: `app.py`
4. Click **"Advanced settings"** → **"Secrets"** and add:

   ```toml
   GROQ_API_KEY = "your_actual_groq_key"
   DATABASE_URL = "postgresql://user:password@host:5432/dbname"
   ```

5. Click **"Deploy"**

Deployment takes 3–5 minutes. Once live, paste the URL into the [Live Demo](#-live-demo) section at the top of this README.

### 3. Supabase free-tier note

Supabase **pauses free projects after 7 days of inactivity**. If the deployed app suddenly stops connecting, restore the project from the Supabase dashboard. For a more pause-proof option, consider [Neon](https://neon.tech).

---

## ⚠️ Known Limitations & Future Work

This is a demo project. The following limitations are **known and intentional**, listed here for transparency.

### Architecture

- **No authentication.** Anyone can query any order by ID. Production systems need customer authentication and order-ownership verification.
- **Human handoff is simulated.** The escalation path returns a canned message but does not integrate with Zendesk, Intercom, Salesforce, email, or WhatsApp.
- **No multi-intent handling.** If the user asks two things at once, the agent asks them to pick one. A production system would handle both.
- **No observability.** No token counting, latency tracking, cost attribution, or tracing (Langfuse / LangSmith).

### RAG

- **Small knowledge base.** Only 3 policy documents. Real deployments need dozens or hundreds with hybrid search, re-ranking, and metadata filtering.
- **Single embedding model.** `all-MiniLM-L6-v2` is fast and free but not the strongest. A production system might use a hosted embedding model with better retrieval quality.
- **No chunking strategy tuning.** Default `RecursiveCharacterTextSplitter` with `chunk_size=500`. Sub-optimal for long documents.

### LLM / Agent

- **Small model.** `gpt-oss-20b` is fast and cheap but occasionally makes classification mistakes on ambiguous inputs.
- **Prompt-engineering-driven classification.** Works well but is not as robust as fine-tuned classifiers.
- **No prompt injection protection.** A malicious user could try to override system instructions.

### Deployment

- **No Docker or CI/CD.** The project runs locally or via Streamlit Cloud. No Dockerfile, no GitHub Actions.
- **Single-process Streamlit.** Not designed for high concurrency.

### Future Work (Prioritized)

1. Add observability (Langfuse) for token, latency, and cost tracking.
2. Integrate real ticketing for human handoff.
3. Add authentication and order-ownership checks.
4. Expand the knowledge base and add hybrid search + re-ranking.
5. Add input guardrails (prompt injection detection, PII redaction).
6. Add Docker + GitHub Actions CI pipeline.

---

## 🔧 Troubleshooting

### App crashes with `psycopg.ProgrammingError: missing "=" ...`

Your `.env` has `postgresql+psycopg://` in `DATABASE_URL`. Remove the `+psycopg`. That suffix is only for SQLAlchemy's internal engine.

### `FATAL: (ENOTFOUND) tenant/user ... not found`

Your Supabase project is either paused, deleted, or the connection string is wrong. Log in to the Supabase dashboard, restore the project, and re-copy the "Session Pooler" URI.

### RAG returns "I don't have enough information" for policy questions

Re-run the ingestion after editing source documents:

```bash
python -m src.rag.ingest
```

The FAISS index must be rebuilt every time `knowledge_base/*.md` changes.

### Groq returns `429 Too Many Requests`

Free-tier Groq has strict rate limits. The app retries automatically with exponential backoff, but the evaluation suite may take 8–10 minutes during peak hours. Reduce load or wait a few minutes.

### Streamlit shows "Sorry, I'm temporarily unable to process your request"

Check the terminal where Streamlit is running — the actual error is logged there with a full traceback.

### `torchvision` warnings spam the terminal

Harmless — Streamlit's file watcher inspects `transformers` vision modules that require `torchvision`. To silence them, add `.streamlit/config.toml`:

```toml
[server]
fileWatcherType = "none"
```

---

## 🔒 Security Notes

- Never commit `.env`. Use `.env.example` as a template.
- Rotate API keys if they are ever exposed.
- The demo database contains shared seed data. Do not use it for anything real.
- Supabase free tier pauses after 7 days of inactivity — restore from the dashboard if the app stops connecting.

---

## 👨‍💻 Author

**Athar Hussain**
AI/ML Engineer | Data Science | LLM Applications

- GitHub: [ATHAR-HUSSAIN-2910](https://github.com/ATHAR-HUSSAIN-2910)
- LinkedIn: [athar-hussain-datascience](https://linkedin.com/in/athar-hussain-datascience)

---

## 📄 License

This project is for educational and portfolio purposes.