# LLM Zoomcamp — Module 4: Evaluation Notes

> Modules 01–03 built RAG pipelines (keyword search, vector search, orchestration) but never measured whether they actually work well — just eyeballed a few example queries. This module builds the missing piece: a ground-truth dataset and metrics to score both **retrieval** (Hit Rate, MRR) and **generation** (an LLM-as-a-judge). All four notebooks chain together: `01` generates the ground truth, `02` scores search against it, `03` generates RAG answers for it, and `04` judges those answers.

---

## Lesson 01 — Generating Ground Truth Data (`01-data-gen.ipynb`)

### Why You Need Ground Truth

You can't measure search or RAG quality without knowing the right answer. Ground truth here means: a set of realistic questions, each labeled with the FAQ record that should answer it. With that in hand, "is this search result good?" becomes "does the returned document's id match the label?" — a yes/no you can average over hundreds of questions.

### Generating Questions with an LLM

For each FAQ record, ask an LLM to emulate a student and write 5 questions that record would answer — using different wording than the record itself, so search can't win by keyword-matching the source text.

```python
data_gen_instructions = """
You emulate a student who's taking our course.
Formulate 5 questions this student might ask based on a FAQ record. The record
should contain the answer to the questions, and the questions should be complete and not too short.
If possible, use as fewer words as possible from the record.

The output should resemble how people ask questions
on the internet. Not too formal, not too short, not too long.
""".strip()
```

### Structured Output with Pydantic

Instead of parsing free-form text, force the model's response into a fixed shape:

```python
from pydantic import BaseModel

class Questions(BaseModel):
    questions: list[str]

response = openai_client.responses.parse(
    model="gpt-5.4-mini",
    input=[
        {"role": "developer", "content": data_gen_instructions},
        {"role": "user", "content": json.dumps(doc)},
    ],
    text_format=Questions,
)

response.output_parsed.questions   # already a list[str], no manual JSON parsing
```

### Reusable Helpers (`zoomcamp/evaluation_utils.py`)

The structured-call pattern above gets reused everywhere in this module, so it's wrapped once:

```python
def llm_structured(client, instructions, user_prompt, output_type, model="gpt-5.4-mini"):
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": user_prompt},
    ]
    response = client.responses.parse(model=model, input=messages, text_format=output_type)
    return response.output_parsed, response.usage
```

`llm_structured_retry` wraps it again with retries and exponential backoff, since structured calls occasionally fail transiently:

```python
def llm_structured_retry(client, instructions, user_prompt, output_type, model="gpt-5.4-mini", max_retries=3):
    for attempt in range(max_retries):
        try:
            return llm_structured(client, instructions, user_prompt, output_type, model=model)
        except Exception:
            if attempt == max_retries - 1:
                raise
            time.sleep(2 ** attempt)
```

### Tracking Cost

`gpt-5.4-mini` bills per million tokens ($0.75 input / $4.50 output). Every LLM call returns a `usage` object; `calc_price(usage)` turns it into a dollar cost, and `calc_total_price(usages)` sums a whole batch:

```python
def calc_price(usage):
    input_cost = (usage.input_tokens / 1_000_000) * 0.75
    output_cost = (usage.output_tokens / 1_000_000) * 4.50
    return {"input_cost": input_cost, "output_cost": output_cost, "total_cost": input_cost + output_cost}
```

### Parallelizing with `map_progress`

79 FAQ records × 1 LLM call each, run sequentially, is slow. `map_progress` fans the calls out across a thread pool while keeping a single `tqdm` progress bar in sync with completed futures:

```python
def map_progress(pool, seq, f):
    results = []
    with tqdm(total=len(seq)) as progress:
        futures = [pool.submit(f, el) for el in seq]
        for future in futures:
            future.add_done_callback(lambda p: progress.update())
        return [future.result() for future in futures]

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, documents, generate_ground_truth)
```

### The Result

79 `llm-zoomcamp` FAQ documents → 395 `(question, document)` ground-truth pairs, generated for **≈$0.057** total, saved to `data/ground_truth.csv`. This file is the input to every other notebook in the module.

---

## Lesson 02 — Evaluating Search Quality (`02-search-eval.ipynb`)

### From "Looks Right" to a Number

Rebuild the same keyword-search index from module 01 (`load_faq_data` + `build_index`). For each ground-truth question, run search and check whether the labeled document comes back — encoded as a 0/1 list, one entry per result position:

```python
def compute_relevance(q, search_function):
    doc_id = q["document"]
    results = search_function(query=q["question"])
    return [int(d["id"] == doc_id) for d in results]
```

`[1, 0, 0, 0, 0]` = correct doc at position 1. `[0, 0, 0, 0, 0]` = not found in the top 5. Do this for all 395 questions and you have the raw material for every metric below.

### Hit Rate

Fraction of questions where the correct document appears *anywhere* in the results — position doesn't matter, only presence:

```python
def hit_rate(relevance):
    return sum(1 for line in relevance if 1 in line) / len(relevance)
```

Baseline keyword search (`question` boost 3.0): **hit_rate ≈ 0.899**.

### Mean Reciprocal Rank (MRR)

Hit Rate can't distinguish a hit at position 1 from a hit at position 5, but users notice. MRR takes `1 / rank` of the first hit for each question (0 if no hit) and averages:

```python
def mrr(relevance):
    total = 0.0
    for line in relevance:
        for rank in range(len(line)):
            if line[rank] == 1:
                total += 1 / (rank + 1)
                break
    return total / len(relevance)
```

Same baseline: **mrr ≈ 0.769** — lower than the hit rate, meaning some hits land below position 1.

### One Function to Score Any Search Variant

```python
def evaluate(ground_truth, search_function):
    relevance_total = compute_relevance_total(ground_truth, search_function)
    return {"hit_rate": hit_rate(relevance_total), "mrr": mrr(relevance_total)}
```

Because `search_function` is a plain `query -> results` callable, `evaluate` works unchanged for keyword search, vector search, hybrid search, or any boost configuration.

### Tuning Boosts by Measuring, Not Guessing

Sweeping the `question` field boost over `[0.5, 1.0, 3.0, 5.0, 10.0]` shows **1.0** beats the original 3.0 (hit_rate 0.924 / mrr 0.814 vs. 0.899 / 0.769) — more weight on one field isn't automatically better. A full grid search over `question`, `answer`, and `section` boosts (36 combinations, each a full 395-question evaluation) pushes the best configuration to **hit_rate ≈ 0.975, mrr ≈ 0.885**.

```python
def search_boosts(query, question_boost, answer_boost, section_boost):
    boost_dict = {"question": question_boost, "answer": answer_boost, "section": section_boost}
    return index.search(query, num_results=5, boost_dict=boost_dict)
```

The pattern generalizes beyond boosts: any change to a search function can be scored by re-running `evaluate` against the same fixed ground truth.

---

## Lesson 03 — Generating RAG Answers for Evaluation (`03-rag-evals.ipynb`)

### Retrieval Isn't the Whole Pipeline

Lesson 02 only scores whether the right *document* comes back. What the user actually sees is the LLM's *generated answer*, and good retrieval doesn't guarantee a good answer — the LLM can still misread the context or add wrong details. This notebook runs the full RAG pipeline over all 395 ground-truth questions and captures both the generated answer and the original FAQ answer, ready for judging in Lesson 04.

### `RAGWithUsage`: RAG Plus Cost Tracking

Subclasses `RAGBase` from module 01 (same `search → build_prompt → llm → rag` pipeline), overriding `llm()` to record every response's `usage`:

```python
class RAGWithUsage(RAGBase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.usages = []

    def llm(self, prompt):
        response = self.llm_client.responses.create(
            model=self.model,
            input=[{"role": "developer", "content": self.instructions},
                   {"role": "user", "content": prompt}],
        )
        self.usages.append(response.usage)
        return response.output_text

    def total_cost(self):
        return calc_total_price(self.usages)
```

`reset_usage()` clears the log — call it right before a batch run so `total_cost()` reflects just that batch, not everything since the assistant was created.

### One Evaluation Record per Question

```python
def generate_rag_answer(rec):
    answer_llm = assistant.rag(rec["question"])
    answer_orig = doc_idx[rec["document"]]["answer"]
    return {
        "question": rec["question"],
        "answer_llm": answer_llm,
        "answer_orig": answer_orig,
        "document": rec["document"],
    }
```

### Running It Over the Full Dataset

Same `ThreadPoolExecutor` + `map_progress` pattern as Lesson 01, now doing a full search-plus-generation round trip per question. 395 questions cost **≈$0.343** in total, saved to `data/rag-answers-new.csv`.

---

## Lesson 04 — LLM-as-a-Judge (`04-llm-judge.ipynb`)

### Why an LLM Judge

Comparing 395 `(answer_llm, answer_orig)` pairs by hand doesn't scale, and exact string matching fails because two answers can be worded completely differently and still be correct. An LLM judge reads both answers and the original question, and decides if they're semantically equivalent.

### A Fair Judge Needs Fair Rules

```python
aqa_judge_instructions = """
You are an expert evaluator. You will be given:
1. A question from a student
2. The original answer from the FAQ (ground truth)
3. An answer generated by an AI assistant

Your task is to decide if the AI answer is semantically equivalent to
the original answer.

Rules:
- The AI answer does NOT need to be word-for-word identical
- It should convey the same key information
- Extra detail is fine as long as the core answer is correct
- Mark 'bad' only if the AI answer is wrong or misses the key point

Be fair and focus on correctness, not style.
""".strip()
```

Without these rules, a strict judge would penalize correct answers just for being phrased differently than the source FAQ.

### Structured Verdict

```python
class AnswerEvaluation(BaseModel):
    reasoning: str = Field(description="Reasoning about the quality of the answer.")
    score: Literal["good", "bad"] = Field(description="'good' if correct and complete, 'bad' otherwise.")
```

Asking for `reasoning` before `score` nudges the model to justify its verdict rather than jumping straight to a label — the same structured-output approach (`llm_structured_retry` + a Pydantic model) used for question generation in Lesson 01, just with a different schema.

### Judging at Scale

```python
def judge_record(rec):
    result, usage = evaluate_aqa(rec["question"], rec["answer_orig"], rec["answer_llm"])
    return {"question": rec["question"], "document": rec["document"],
            "score": result.score, "reasoning": result.reasoning}, usage

with ThreadPoolExecutor(max_workers=6) as pool:
    results = map_progress(pool, answers, judge_record)
```

Judging all 395 answers costs **≈$0.251** — more than generating the questions ($0.057), since each judge call reads a full question plus two answers as input.

### The Result

```
score
good    379   (≈95.9%)
bad      16   (≈4.1%)
```

Filtering to `df_eval[df_eval["score"] == "bad"]` surfaces the actual failures with the judge's reasoning attached (e.g. GPU-hours questions, peer-review requirements, model-selection details) — a concrete debugging list, not just a single opaque score. Results are saved to `data/rag-evaluations-new.csv`.

---

## Key Concepts Summary

| Concept | One-line summary |
|---|---|
| Ground truth | Question → correct document/answer pairs, generated by an LLM emulating a student |
| Structured output | `responses.parse(text_format=PydanticModel)` — no manual JSON parsing |
| `llm_structured` / `llm_structured_retry` | Reusable structured-call helpers with retry + backoff |
| Relevance list | Per-question 0/1 list marking which result position (if any) matched the ground truth |
| Hit Rate | Fraction of questions where the correct document appears anywhere in the results |
| MRR | Average of `1/rank` of the first hit — rewards finding the right answer *early* |
| `evaluate(ground_truth, search_function)` | One function, any search variant — the basis for boost tuning |
| `RAGWithUsage` | `RAGBase` + per-call cost tracking, so full pipeline runs can be priced |
| LLM-as-a-judge | An LLM scores semantic equivalence between generated and ground-truth answers |
| `map_progress` | `ThreadPoolExecutor` + `tqdm`, used throughout to parallelize LLM calls |

## The Evaluation Chain Across Lessons

```
01  Ground truth generation   → 395 (question, document) pairs           ($0.057)
02  Search evaluation         → Hit Rate / MRR, boost tuning             (0.899 → 0.975 hit rate)
03  RAG answer generation     → 395 (answer_llm, answer_orig) pairs      ($0.343)
04  LLM-as-a-judge            → good/bad verdict per answer              ($0.251, 95.9% good)
```

Each stage measures a different layer of the pipeline: 02 checks retrieval alone, 03 produces what the user actually sees, and 04 scores whether that output is actually correct — three separate numbers instead of one gut feeling.
