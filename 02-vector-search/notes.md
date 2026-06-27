# LLM Zoomcamp — Module 2: Vector Search Notes

> Module 01 used **keyword search** (minsearch / sqlitesearch). This module replaces it with **vector search** — encoding text as numbers and finding semantic neighbors. The RAG pipeline structure stays identical; only the retrieval backend changes.

---

## Lesson 01 — Vector Search (`01-vector-search.ipynb`)

### The Problem with Keyword Search

Keyword search matches on exact words. Ask "Can I still enroll?" and it won't find "Can I join the course?" unless both share keywords. Semantic meaning is lost.

Vector search fixes this: sentences with the same *meaning* get similar vectors, even when the wording is completely different.

### Sentence Embeddings

A sentence embedding model converts text into a fixed-size numerical vector. Similar sentences map to nearby vectors; unrelated sentences map to distant vectors.

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')
# produces 384-dimensional vectors

q1 = "I just discovered the course, can I still join?"
q2 = "I just found out about the program, can I still enroll?"

v1 = model.encode(q1)   # shape: (384,)
v2 = model.encode(q2)   # shape: (384,)
```

`all-MiniLM-L6-v2` is a compact, fast model — 384 dimensions, runs on CPU, good quality for retrieval tasks.

### Similarity via Dot Product

The dot product between two unit vectors equals their cosine similarity. Higher = more similar. Range is roughly 0 to 1.

```python
v1.dot(v2)    # → 0.94  (very similar wording)
v1.dot(dv)    # → 0.40  (related topic)
v3.dot(dv)    # → 0.02  (unrelated topic)
```

No library needed for this — it's just NumPy math.

---

### Brute-Force Search Over a Corpus

Encode all documents into a matrix, then a single matrix multiply scores all of them at once.

**Step 1 — Build the embedding matrix:**

```python
documents = load_faq_data()  # 1350 Q&A documents

texts = [doc["question"] + " " + doc["answer"] for doc in documents]

batch_size = 50
vectors = []

for i in tqdm(range(0, len(texts), batch_size)):
    batch = texts[i:i + batch_size]
    batch_vectors = model.encode(batch)
    vectors.extend(batch_vectors)

X = np.array(vectors)  # shape: (1350, 384)
```

Concatenating question + answer gives the model more signal than either field alone.

**Step 2 — Score and rank:**

```python
query = "Can I still join the course after the start date?"
v_query = model.encode(query)

scores = X.dot(v_query)           # shape: (1350,) — one score per doc
idx = np.argmax(scores)           # index of the best match
top5 = np.argsort(scores)[-5:]    # indices of top-5 matches
```

`X.dot(v_query)` is O(n × d) — linear in the number of documents. Fast enough for thousands of docs; needs an index at millions.

---

### `VectorSearch` from `minsearch`

Wraps the brute-force approach in a clean API. Same interface as keyword `Index` — you just pass a vector instead of a query string.

```python
from minsearch import VectorSearch

vindex = VectorSearch(keyword_fields=["course"])
vindex.fit(X, documents)

query_vector = model.encode("I just discovered the course. Can I still join it?")

# Search all courses
results = vindex.search(query_vector, num_results=5)

# Filter to one course before ranking
results = vindex.search(
    query_vector,
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

`filter_dict` narrows candidates first, then ranks — not post-filter. More efficient and gives better results within the target course.

---

### RAG Baseline: Keyword Retrieval

`RAGBase` from the project package wires any index to an LLM. The keyword index from module 01 is the baseline:

```python
from zoomcamp.rag_helper import RAGBase

documents = load_faq_data()
index = build_index(documents)  # minsearch keyword index

assistant = RAGBase(
    index=index,
    llm_client=openai_client,
    model="gpt-4o-mini"
)

assistant.rag("I just found out about the program, can I still sign up?")
# → "Yes, but to receive a certificate..."
```

---

### `RAGVector` — Swapping in Vector Retrieval

Subclass `RAGBase` and override only `.search()`. Everything else — prompt building, LLM call, response parsing — is inherited unchanged.

```python
class RAGVector(RAGBase):

    def __init__(self, embedder, **kwargs):
        super().__init__(**kwargs)
        self.embedder = embedder

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        filter_dict = {"course": self.course}

        return self.index.search(
            query_vector,
            num_results=num_results,
            filter_dict=filter_dict
        )
```

```python
vector_assistant = RAGVector(
    embedder=model,
    index=vindex,
    llm_client=openai_client,
    model="gpt-4o-mini"
)

vector_assistant.rag("the program has already begun, can I still sign up?")
# → "Yes, you can still join..."
```

**Why subclass instead of rewrite?** RAG has five moving parts (search, build context, build prompt, call LLM, parse response). Only one changes — subclassing keeps the diff minimal.

---

### Persistent Vector Index with `sqlitesearch`

`minsearch.VectorSearch` lives in memory and disappears on restart. `sqlitesearch.VectorSearchIndex` persists to a SQLite file with an IVF (inverted file) approximate nearest-neighbor index.

```python
from sqlitesearch import VectorSearchIndex

vs_index = VectorSearchIndex(
    keyword_fields=['course'],
    mode='ivf',             # approximate nearest-neighbor
    db_path='db/faq_vectors.db'
)

# Same search API as minsearch
results = vs_index.search(
    query_vector,
    filter_dict={'course': 'llm-zoomcamp'},
    num_results=5
)

vs_index.close()  # flush and release the file handle
```

`mode='ivf'` clusters vectors at build time and searches only the nearest cluster at query time — sub-linear but slightly approximate. For this dataset size, exact search is fine too.

---

## Lesson 02 — pgvector (`02-vector-search-pgvector.ipynb`)

### Why Move to PostgreSQL?

Both `minsearch` and `sqlitesearch` are single-process stores. PostgreSQL + pgvector gives you:

- **Concurrent access** — multiple processes/services query the same index
- **SQL joins and filters** — combine vector search with any relational query
- **Operational familiarity** — same DB as the rest of your stack, same backups/monitoring
- **HNSW index** — production-grade approximate nearest-neighbor built in

---

### Connecting and Enabling pgvector

```python
import psycopg

conn = psycopg.connect('postgresql://user:pswd@localhost:5432/faq')
conn.execute("CREATE EXTENSION IF NOT EXISTS vector")
```

`CREATE EXTENSION vector` installs pgvector once per database. It adds:
- The `vector(n)` column type
- Distance operators: `<=>` (cosine), `<->` (L2), `<#>` (negative inner product)

---

### Schema: `vector(384)` Column

```python
conn.execute("""
    CREATE TABLE documents (
        id        SERIAL PRIMARY KEY,
        course    TEXT,
        section   TEXT,
        question  TEXT,
        answer    TEXT,
        embedding vector(384)
    )
""")
```

`vector(384)` enforces the dimension at the DB level — inserting a 512-dim vector raises an error immediately, not silently at query time.

---

### Inserting Embeddings

pgvector's `::vector` cast accepts the `[x,y,z,...]` string literal. A helper serializes NumPy arrays to that format:

```python
def vec_to_str(vector):
    return '[' + ','.join(str(x) for x in vector) + ']'

for doc, vec in tqdm(zip(documents, vectors), total=len(documents)):
    conn.execute(
        """
        INSERT INTO documents (course, section, question, answer, embedding)
        VALUES (%s, %s, %s, %s, %s::vector)
        """,
        (doc['course'], doc['section'], doc['question'], doc['answer'],
         vec_to_str(vec))
    )

conn.commit()
```

Commit once at the end — wrapping 1350 inserts in a single transaction is much faster than auto-committing each row.

---

### Cosine Similarity Search with SQL

```python
query = 'I just discovered the course. Can I still join it?'
query_str = vec_to_str(model.encode(query))

results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, query_str)
).fetchall()
```

Key points:
- `<=>` is **cosine distance** (0 = identical, 2 = opposite). Lower = more similar.
- `1 - (embedding <=> query)` converts to a **similarity score** (higher = more similar).
- `ORDER BY embedding <=> query` ranks rows by distance — pgvector handles this natively; no application-side sorting needed.
- The query vector appears twice because SQL doesn't allow column aliases in `WHERE`/`ORDER BY` that reference `SELECT` expressions.

---

### Filtered Search by Course

```python
results = conn.execute(
    """
    SELECT course, question, answer,
           1 - (embedding <=> %s::vector) AS similarity
    FROM documents
    WHERE course = %s
    ORDER BY embedding <=> %s::vector
    LIMIT 5
    """,
    (query_str, 'llm-zoomcamp', query_str)
).fetchall()
```

`WHERE course = %s` reduces the candidate set before ranking. This is the SQL equivalent of `filter_dict` in minsearch/sqlitesearch. It's important for multi-course setups — without it, results from all courses compete.

---

### HNSW Approximate Nearest-Neighbor Index

Without an index, every query scans all rows — exact but O(n). At thousands of documents this is fast; at millions it becomes a bottleneck.

```python
conn.execute("""
    CREATE INDEX ON documents
    USING hnsw (embedding vector_cosine_ops)
""")
```

HNSW (Hierarchical Navigable Small World) builds a multi-layer graph of vectors. Queries traverse the graph instead of scanning every row — typically O(log n).

`vector_cosine_ops` must match the operator used in queries (`<=>`). Using the wrong ops type will silently skip the index.

**Accuracy trade-off:** HNSW is approximate — it may miss the true nearest neighbor in rare cases. For RAG, this is almost always acceptable: slightly imperfect retrieval barely affects answer quality.

---

### `pgvector_search` Helper

Encapsulates the full encode → serialize → SQL query → dict pipeline:

```python
def pgvector_search(query, course='llm-zoomcamp', num_results=5):
    query_vector = model.encode(query)
    query_str = vec_to_str(query_vector)

    rows = conn.execute(
        """
        SELECT course, section, question, answer
        FROM documents
        WHERE course = %s
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (course, query_str, num_results)
    ).fetchall()

    return [
        {'course': r[0], 'section': r[1], 'question': r[2], 'answer': r[3]}
        for r in rows
    ]
```

Returns the same list-of-dicts shape as `minsearch.VectorSearch.search()` and `sqlitesearch.VectorSearchIndex.search()` — drop-in compatible with `RAGBase`.

---

### `RAGPgVector` — RAG Backed by pgvector

Same pattern as `RAGVector` from Lesson 01: subclass `RAGBase`, override only `.search()`.

```python
from zoomcamp.rag_helper import RAGBase

class RAGPgVector(RAGBase):

    def __init__(self, embedder, conn, **kwargs):
        super().__init__(index=None, **kwargs)
        self.embedder = embedder
        self.conn = conn

    def search(self, query, num_results=5):
        query_vector = self.embedder.encode(query)
        query_str = vec_to_str(query_vector)

        rows = self.conn.execute(
            """
            SELECT course, section, question, answer
            FROM documents
            WHERE course = %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            (self.course, query_str, num_results)
        ).fetchall()

        return [
            {'course': r[0], 'section': r[1], 'question': r[2], 'answer': r[3]}
            for r in rows
        ]
```

```python
vector_assistant = RAGPgVector(
    embedder=model,
    conn=conn,
    llm_client=openai_client,
)

vector_assistant.rag("the program has already begun, can I still sign up?")
# → "Yes, you can still join..."
```

`index=None` is passed to `RAGBase.__init__` because the DB connection replaces the index. The LLM call, prompt building, and response parsing are all inherited — the only change is where retrieval happens.

---

## Key Concepts Summary

| Concept | One-line summary |
|---|---|
| Sentence embedding | Maps text to a fixed-size vector; semantically similar text → nearby vectors |
| `all-MiniLM-L6-v2` | Fast 384-dim model; good quality for retrieval, runs on CPU |
| Dot product similarity | `v1.dot(v2)` measures cosine similarity for unit vectors |
| Brute-force search | `X.dot(v_query)` scores all docs at once with a single matrix multiply |
| `minsearch.VectorSearch` | In-memory vector search with keyword pre-filter; disappears on restart |
| `sqlitesearch.VectorSearchIndex` | Disk-persisted vector search with IVF approximate index |
| pgvector `<=>` | SQL cosine distance operator; lower = more similar |
| `vector(384)` column | Postgres type that stores embeddings and enforces dimension |
| HNSW index | Approximate nearest-neighbor graph; sub-linear queries, slight accuracy trade-off |
| `RAGVector` / `RAGPgVector` | Subclass `RAGBase`, override only `.search()`; inherit everything else |
| `vec_to_str` | Serializes NumPy array to `[x,y,...]` string for pgvector's `::vector` cast |

---

## The Evolution of the Retrieval Backend

```
Module 01  minsearch keyword Index     — in-memory, BM25-style, exact match
Module 01  sqlitesearch TextSearchIndex — disk-persisted keyword search

Module 02  minsearch VectorSearch      — in-memory, cosine similarity (NumPy)
Module 02  sqlitesearch VectorSearchIndex — disk-persisted, IVF approximate
Module 02  PostgreSQL + pgvector       — concurrent, SQL-native, HNSW index
```

The RAG pipeline (`RAGBase`) never changes — only the class returned by `.search()` does.
