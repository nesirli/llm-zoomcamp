# LLM Zoomcamp — Module 5: Monitoring Notes

> Modules 01–04 built and evaluated a RAG pipeline offline, in notebooks. This module puts it behind a real UI and watches it in production: every answer is timed, priced, and judged automatically; users can thumbs up/down it; and everything lands in Postgres so a Streamlit dashboard (and later Grafana) can chart it. No notebooks this module — it's a small service made of plain Python scripts, wired together with Docker Compose.

---

## Instrumenting the RAG Pipeline (`metrics.py`)

### The Problem: `RAGBase` Only Returns Text

`RAGBase.rag()` (from `zoomcamp.rag_helper`, built in earlier modules) returns just the final answer string. To monitor anything — cost, latency, token usage — that data has to be captured at the point of the LLM call, not reconstructed afterward.

### `RAGWithMetrics`: Same Subclass-and-Override Pattern as Module 02

Same trick as `RAGVector`/`RAGPgVector` in Module 02: subclass `RAGBase`, override only `.llm()`, keep search/prompt-building/answer-parsing untouched.

```python
class RAGWithMetrics(RAGBase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.last_call: LLMCallRecord = None

    def llm(self, prompt):
        start_time = time.time()
        response = self._call_llm(prompt)
        response_time = time.time() - start_time
        self._log_response(prompt, response, response_time)
        return response.output_text
```

`self.last_call` holds the most recent `LLMCallRecord` — the caller (the Streamlit app) reads it right after `assistant.rag(...)` returns instead of threading extra return values through `RAGBase`.

### `LLMCallRecord`: One Dataclass, One Row

```python
@dataclass
class LLMCallRecord:
    model: str
    prompt: str
    instructions: str
    answer: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    response_time: float
    cost: float
    timestamp: datetime = field(default_factory=datetime.now)
```

This shape is reused everywhere: it's what gets built after each LLM call, what `db_save.save_conversation` writes to Postgres, and what `db_query.get_conversations` reconstructs rows back into.

### Pricing Is Just Arithmetic on `usage`

```python
def calculate_cost(model, usage):
    cost = 0
    if "gpt-5.4-mini" in model:
        cost = (usage.input_tokens * 0.15 + usage.output_tokens * 0.60) / 1_000_000
    return cost
```

Same idea as `calc_price` in Module 04's evaluation utils — per-token rates, applied to the `usage` object every OpenAI response already carries. No external cost-tracking service needed for a single-model app.

---

## Persisting Everything in Postgres (`db_init.py`, `db_save.py`, `db_feedback.py`)

### Two Tables: What Happened, and What People Thought of It

`db_init.py` creates two tables:

- **`conversations`** — one row per question asked: the question, answer, model, full prompt/instructions, token counts, response time, cost, timestamp.
- **`feedback`** — one row per judgment on a conversation, linked by `conversation_id`. `source` distinguishes who judged: `"judge"` (LLM) rows carry `relevance` + `explanation`; `"user"` (thumbs up/down) rows carry `score`.

```python
cur.execute("""
    CREATE TABLE IF NOT EXISTS feedback (
        id SERIAL PRIMARY KEY,
        conversation_id INTEGER REFERENCES conversations(id),
        source TEXT NOT NULL,
        relevance TEXT,
        explanation TEXT,
        score INTEGER,
        timestamp TIMESTAMP WITH TIME ZONE NOT NULL
    )
""")
```

One table, two feedback sources, distinguished by a column rather than two separate schemas — keeps every downstream query (`get_relevance_stats`, `get_user_feedback_stats`) a single `GROUP BY source`/`WHERE source = ...`.

### Connection-per-call, Not a Pool

```python
def get_db_connection():
    return psycopg.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        dbname=os.getenv("POSTGRES_DB", "course_assistant"),
        user=os.getenv("POSTGRES_USER", "user"),
        password=os.getenv("POSTGRES_PASSWORD", "password"),
    )
```

Every `save_conversation`, `save_feedback`, and query function opens a connection, does its work in a `try`/`finally`, and closes it. Fine at Streamlit-app scale (one request at a time); a connection pool would be the next step under real concurrent load.

### Saving Returns the ID You Need Next

```python
cur.execute(
    """
    INSERT INTO conversations (...) VALUES (...)
    RETURNING id
    """,
    (...),
)
conversation_id = cur.fetchone()[0]
```

`RETURNING id` hands back the new row's primary key in the same round trip, so the app can immediately attach judge and user feedback to that exact conversation without a second lookup.

### Fixed Timezone at Import Time

```python
DB_TIMEZONE = datetime.now().astimezone().tzinfo
```

Computed once when `db_init` is imported, then reused by both `save_conversation` and `save_feedback` so every timestamp in a given run is consistently tagged with the same offset.

---

## The Chat App (`app.py`)

Every "Ask" click does four things in sequence, each building on the last:

```python
answer = assistant.rag(user_input)                    # 1. generate the answer
record = assistant.last_call                          # 2. pull the metrics captured by RAGWithMetrics
conversation_id = save_conversation(record, user_input, "llm-zoomcamp")   # 3. persist it
relevance, explanation = evaluate_relevance(user_input, answer)           # 4. auto-judge it
save_feedback(conversation_id, "judge", relevance=relevance, explanation=explanation)
```

The response time, token counts, and cost pulled from `record` are shown directly in the UI — monitoring isn't just for the dashboard, it's visible to the user asking the question.

### Thumbs Up/Down Feeds the Same Table as the Judge

```python
if st.button("+1"):
    cid = st.session_state.conversation_id
    save_feedback(cid, "user", score=1)
```

`st.session_state.conversation_id` is what makes this work — it's set once per question (step 3 above) and stays around for the feedback buttons rendered afterward, so a thumbs up/down always attaches to the conversation the user just saw, not whatever was asked last.

---

## The Monitoring Dashboard (`dashboard.py`)

A second, separate Streamlit page reads the same database `app.py` writes to — no shared process, no in-memory state, just SQL as the interface between the "chat" surface and the "monitoring" surface.

```python
stats = get_stats()
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total conversations", stats.total)
col2.metric("Avg response time", f"{stats.avg_response_time:.2f}s")
col3.metric("Total cost", f"${stats.total_cost:.4f}")
col4.metric("Avg tokens", f"{stats.avg_tokens:.0f}")
```

Four aggregate SQL queries in `db_query.py` (`get_stats`, `get_conversations`, `get_relevance_stats`, `get_user_feedback_stats`) drive the whole page — the dashboard has no logic of its own, it just renders what SQL already computed:

```python
def get_relevance_stats():
    cur.execute("""
        SELECT relevance, COUNT(*)
        FROM feedback
        WHERE source = 'judge'
        GROUP BY relevance
    """)
    return dict(rows)
```

`st.line_chart(df, x="timestamp", y="cost")` and `st.bar_chart(relevance)` turn those aggregates into charts with no extra plotting code — Streamlit infers the chart type from the DataFrame shape.

---

## Online LLM-as-a-Judge (`judge.py`)

### Different from Module 04's Judge, Same Machinery

Module 04's judge compared a generated answer against a *known-correct* answer, offline, over a fixed dataset of 395 questions. This judge has no ground truth to compare against — it runs online, once per live question, and just asks: does this answer address this question at all?

```python
class RelevanceVerdict(BaseModel):
    relevance: Literal["NON_RELEVANT", "PARTLY_RELEVANT", "RELEVANT"]
    explanation: str
```

Three-way relevance instead of Module 04's binary `good`/`bad` — appropriate for judging *without* a reference answer, where "partially addressed the question" is a real, common outcome rather than a forced binary call.

Reuses `llm_structured_retry` from `zoomcamp.evaluation_utils` (Module 04) unchanged — the retry-with-backoff wrapper around `responses.parse` doesn't care what Pydantic model or prompt it's given.

```python
def evaluate_relevance(question, answer, client=None):
    ...
    result, usage = llm_structured_retry(client, judge_instructions, prompt, RelevanceVerdict)
    return result.relevance, result.explanation
```

Called synchronously inside the Streamlit request in `app.py` — the user waits an extra LLM round trip per question so the relevance verdict is ready to display immediately, trading a bit of latency for judging every single production answer instead of just a sampled batch.

---

## Generating Synthetic Traffic (`generate_data.py`)

A dashboard with zero rows is impossible to evaluate. `generate_data.py` fabricates plausible-looking `LLMCallRecord`s (random tokens, cost, response time from a fixed pool of Q&A pairs) and writes them straight to Postgres through the same `save_conversation`/`save_feedback` functions the real app uses — no separate "fake data" code path, no OpenAI calls, no cost.

```python
def generate_one():
    question = random.choice(SAMPLE_QUESTIONS)
    answer = random.choice(SAMPLE_ANSWERS)
    record = fake_record(question, answer)
    conversation_id = save_conversation(record, question, "llm-zoomcamp")

    if random.random() < 0.7:
        relevance = random.choice(RELEVANCE)
        save_feedback(conversation_id, "judge", relevance=relevance, ...)
    if random.random() < 0.5:
        save_feedback(conversation_id, "user", score=random_score())
```

`generate_live()` loops `generate_one()` every second so the dashboard's time-series charts have a continuous stream to plot rather than a single static batch — closer to what a real traffic pattern looks like.

---

## Running It: Docker Compose Stack

Three services, one shared `.env`:

```yaml
services:
  postgres:    # image: postgres:17 — stores conversations + feedback
  grafana:     # image: grafana/grafana — points at the same Postgres for richer dashboards
  streamlit:   # built from the local Dockerfile — runs app.py
```

`streamlit` depends on `postgres` and gets `POSTGRES_HOST=postgres` (the Compose service name, not `localhost`) — the same `get_db_connection()` code from `db_init.py` works both on a laptop (`localhost` default) and inside Compose (service-name override) purely through env vars.

```dockerfile
COPY pyproject.toml uv.lock .python-version ./
RUN uv sync --locked
COPY . .
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

Dependencies are installed from the lockfile *before* the app code is copied in, so rebuilding after an app-only change reuses Docker's layer cache instead of resolving `uv.lock` again. `--server.address=0.0.0.0` is required inside a container — Streamlit's default `localhost` binding wouldn't be reachable from outside.

Grafana here is set up as a second consumer of the same Postgres data the Streamlit dashboard already reads — the point being that once metrics land in a real database, any BI tool can plug into them, not just the one you wrote yourself.

---

## Key Concepts Summary

| Concept | One-line summary |
|---|---|
| `RAGWithMetrics` | Subclasses `RAGBase`, overrides only `.llm()`, to time and price every call |
| `LLMCallRecord` | One dataclass shared by the metrics capture, the DB write, and the DB read |
| `conversations` / `feedback` tables | What happened vs. what people (or the judge) thought of it, linked by `conversation_id` |
| `source` column (`"judge"` vs `"user"`) | One feedback table serves two very different feedback producers |
| `RETURNING id` | Get the new row's PK in the same insert, needed to attach feedback right after |
| Online judge (`judge.py`) | Three-way relevance verdict with no reference answer, run per live question |
| Offline judge (Module 04) | Binary good/bad verdict against a known-correct answer, run over a fixed batch |
| Streamlit chat + dashboard | Two separate pages, same Postgres — SQL is the interface between them |
| `generate_data.py` | Synthetic traffic through the real save functions, so the dashboard has data to show |
| Docker Compose (postgres + grafana + streamlit) | `POSTGRES_HOST` env var is the only thing that changes between local and containerized runs |

## From Offline Evaluation to Online Monitoring

```
Module 04   Fixed 395-question dataset  → batch judge → single accuracy number, computed once
Module 05   Live user questions         → per-request judge + user feedback → time series in a dashboard
```

Module 04 answers "is this RAG pipeline good, on average, right now?" Module 05 answers "is it *staying* good, question by question, as real people use it?" — the judge logic barely changes, but running it online, per request, and persisting every verdict is what turns evaluation into monitoring.
