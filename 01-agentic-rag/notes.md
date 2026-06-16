# LLM Zoomcamp — Module 1: Agentic RAG Notes

> Early lessons (01–05) use **DeepSeek** via `client.chat.completions.create`.
> The API is OpenAI-compatible, so the patterns are identical.
> Lesson 06 switches to **OpenAI** via `openai_client.responses.create` (a newer API).
> These notes explain everything in the OpenAI context.

---

## Lesson 01 — RAG Starter

### The Problem: LLM Knows Nothing About Your Course

An LLM is trained on general internet data. Ask it a course-specific question and it either makes something up or says "I don't know which course you mean."

```python
question = "I just discovered the course. Can I join now?"
answer = llm(question)
# → "Could you tell me which course you're referring to?"
```

### The Fix: Give It Context Manually

You paste relevant facts directly into the prompt. The LLM now answers from those facts instead of guessing.

```python
context = """
I just discovered the course. Can I still join?
Yes, but if you want to receive a certificate, you need to submit your project...
"""

prompt = f"""
Answer the question using this context.

Question: {question}
Context: {context}
"""

answer = llm(prompt)
# → "Yes, you can join. But to get a certificate, submit your project..."
```

### The RAG Pattern

RAG = **Retrieval Augmented Generation**. Three steps every time:

```
1. Search  →  find relevant documents from your knowledge base
2. Build prompt  →  inject those documents into the LLM prompt as context
3. LLM  →  generate an answer grounded in that context
```

```python
def rag(question):
    search_results = search(question)       # step 1
    prompt = build_prompt(question, search_results)  # step 2
    return llm(prompt)                      # step 3
```

---

## Lesson 02 — RAG Complete with minsearch

### Loading Documents

Documents come from the DataTalks.Club FAQ API — a JSON list of Q&A pairs, each tagged with a `course` field.

```python
import requests
response = requests.get("https://datatalks.club/faq/json/courses.json")
# → list of 1342 documents across 6 courses
```

Each document looks like:
```python
{
    "id": "74eb249bbf",
    "course": "llm-zoomcamp",
    "section": "General Course-Related Questions",
    "question": "I just discovered the course. Can I still join?",
    "answer": "Yes, but if you want to receive a certificate..."
}
```

### In-Memory Search with minsearch

`minsearch` is a simple keyword search library. You build an index in memory and search it.

```python
from minsearch import Index

index = Index(
    text_fields=["section", "question", "answer"],  # searchable text
    keyword_fields=["course"]                        # exact-match filters
)
index.fit(documents)
```

**Boosting** — give higher weight to matches in the `question` field (more relevant) and lower to `section`:

```python
results = index.search(
    query,
    boost_dict={"question": 2.0, "section": 0.5},
    filter_dict={"course": "llm-zoomcamp"},
    num_results=5
)
```

### Building the Prompt

You turn search results into a readable context block, then inject it into the prompt.

```python
INSTRUCTIONS = """
Your task is to answer questions from course participants
based on the provided context. If the answer is not in the context,
respond with "I don't know."
"""

def build_context(search_results):
    lines = []
    for doc in search_results:
        lines.append(doc["section"])
        lines.append("Q: " + doc["question"])
        lines.append("A: " + doc["answer"])
        lines.append("")
    return "\n".join(lines).strip()

def build_prompt(question, search_results):
    context = build_context(search_results)
    return f"Question:\n{question}\n\nContext:\n{context}"
```

### Calling the LLM (OpenAI-compatible API)

```python
def llm(instructions, user_prompt, model="gpt-4o-mini"):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": instructions},
            {"role": "user", "content": user_prompt},
        ]
    )
    return response.choices[0].message.content
```

Two roles:
- `"system"` — the developer's instructions (role, behavior, constraints)
- `"user"` — the actual question + context

### Token Usage and Cost

Every API call reports how many tokens were used:

```python
response.usage
# → CompletionUsage(prompt_tokens=337, completion_tokens=140, total_tokens=477)

cost = (prompt_tokens * input_price + completion_tokens * output_price)
```

Prompt tokens (input) are cheaper than completion tokens (output). Longer contexts = more input tokens = higher cost.

### The RAGBase Class

The course wraps the whole pipeline into a reusable class:

```python
class RAGBase:
    def __init__(self, index, llm_client, ...):
        self.index = index
        self.llm_client = llm_client

    def search(self, query): ...
    def build_context(self, results): ...
    def build_prompt(self, query, results): ...
    def llm(self, prompt): ...
    def rag(self, query):
        results = self.search(query)
        prompt = self.build_prompt(query, results)
        return self.llm(prompt)
```

---

## Lesson 03 — SQLite Ingest

### Why Move from minsearch to SQLite?

`minsearch` keeps the index in memory — it disappears when the process ends. You'd have to re-index every time. SQLite persists the index to a file on disk.

### Building a Persistent Index with sqlitesearch

```python
from sqlitesearch import TextSearchIndex

index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="db/faq.db"      # saved to disk
)

for doc in docs_llm:
    index.add(doc)

index.close()
```

Next time you start, you just open the file — no re-indexing needed:

```python
index = TextSearchIndex(
    text_fields=["question", "section", "answer"],
    keyword_fields=["course"],
    db_path="db/faq.db"      # reads existing file
)
```

---

## Lesson 04 — RAG Complete with SQLite Search

Same RAG pipeline as Lesson 02, but using the SQLite index instead of minsearch. The `RAGBase` class works with any index that has a `.search()` method — you just swap it out.

```python
sqlite_index = TextSearchIndex(..., db_path="db/faq.db")

assistant = RAGBase(
    index=sqlite_index,
    llm_client=client,
)

answer = assistant.rag("I just discovered the course. Can I join now?")
```

---

## Lesson 05 — Function Calling

### The Problem with Fixed RAG

In the RAG pipeline, **the LLM is a passenger**. You run the search for it, build the prompt, and hand it a finished package. The LLM has no say in what gets searched or how.

This breaks on typos, ambiguous queries, or questions that need multiple searches. The pipeline doesn't recover — it always runs exactly one search and hands over whatever it found.

### Function Calling: The LLM Drives

Instead of running search yourself, you **tell the LLM that a search function exists**. It decides when to call it and what to search for. If the results are bad, it can try again on its own.

### Step 1 — Define the Tool as JSON

You describe the function to the LLM in JSON. The LLM never sees your Python code — it's language-agnostic (HTTP calls under the hood).

```python
search_tool = {
    "type": "function",
    "function": {
        "name": "search",
        "description": "Search the FAQ database for entries matching the given query.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Search query text to look up in the course FAQ."
                }
            },
            "required": ["query"],
            "additionalProperties": False
        }
    }
}
```

The `description` field is the most important part — the LLM reads it to decide when to call the function.

### Step 2 — First API Call: LLM Decides to Search

```python
messages = [
    {"role": "user", "content": "I just discovered the course. Can I join it?"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[search_tool],
)

print(response.choices[0].message.tool_calls)
# → [tool_call(name="search", arguments='{"query": "Can I join the course late"}')]
```

The LLM doesn't give you an answer. It says "I want to call `search` with these arguments." Notice it rewrites the query — better search keywords, not the user's exact words.

### Step 3 — Execute the Function

The LLM told you what to call. Now you actually run it:

```python
import json

tool_call = response.choices[0].message.tool_calls[0]
args = json.loads(tool_call.function.arguments)   # parse JSON → dict

results = search(**args)            # run the actual Python function
result_json = json.dumps(results)   # serialize back to JSON string
```

### Step 4 — Send Everything Back

Append two things to the message history, then make a second API call:

```python
# 1. The model's tool-call message (model needs to see its own decision)
messages.append(response.choices[0].message)

# 2. The tool result, linked by tool_call_id
messages.append({
    "role": "tool",
    "tool_call_id": tool_call.id,    # links result to the specific call
    "content": result_json,
})

# Second call: model now has question + its tool call + FAQ results
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=[search_tool],
)

print(response.choices[0].message.content)
# → "Yes, you can still join! However, to get a certificate..."
```

**Why send the full history?** LLMs are stateless between API calls. Each call starts fresh. The entire memory — question, tool call, tool result — must be resent every time.

### The Two-Call Pattern

```
Call 1: question → model says "call search("...")"
  ↓ you run search, get results
Call 2: question + tool_call + results → model gives final answer
```

Two API calls = two billing events. The second is more expensive because it resends everything as input.

---

## Lesson 06 — The Agentic Loop

### Why One Round-Trip Is Not Enough

Function calling by hand breaks when the LLM needs more than one search. You don't know in advance how many searches it will want — the LLM decides based on what it finds. So you need a loop that keeps calling the LLM and running tools until it's done.

### What Is an Agent?

An agent = an LLM in a loop with tools.

| Part | What it is | In code |
|---|---|---|
| **Instructions** | Role and rules for the LLM | `{"role": "developer", "content": instructions}` |
| **Tools** | Functions the LLM can call | `tools=[search_tool]` |
| **Memory** | Full message history | The `messages` list, appended every iteration |

> Note: The agentic loop lesson uses `openai_client.responses.create` (OpenAI's newer Responses API) instead of `chat.completions.create`. The tool schema format is slightly different — no nested `"function"` key.

### Instructions (Developer Prompt)

The developer prompt tells the agent how to behave. This is the primary way to steer an agent.

```python
instructions = """
You're a course teaching assistant.
Answer questions from course students.

Use the search function to look up information.
Use as many keywords from the user question as possible.
Make multiple searches. First search, analyze results, then search more.

At the end, ask if there are other areas the user wants to explore.
""".strip()
```

### Tool Definition (Responses API format)

```python
search_tool = {
    "type": "function",
    "name": "search",                  # no nested "function" key here
    "description": "Search the FAQ database for entries matching the given query.",
    "parameters": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Search query text to look up in the course FAQ."
            }
        },
        "required": ["query"],
        "additionalProperties": False
    }
}
```

### The `make_call` Helper

A helper that parses the LLM's tool call, runs the function, and returns the result in the format the Responses API expects:

```python
def make_call(call):
    args = json.loads(call.arguments)   # parse JSON string → dict

    if call.name == "search":
        result = search(**args)

    result_json = json.dumps(result, indent=2)

    return {
        "type": "function_call_output",
        "call_id": call.call_id,        # must match the LLM's call_id
        "output": result_json,
    }
```

### The Agentic Loop

```python
it = 1

while True:
    print(f"iteration #{it}...")
    has_function_calls = False          # reset every iteration

    response = openai_client.responses.create(
        model="gpt-4o-mini",
        input=messages,                 # full history every time
        tools=[search_tool],
    )

    messages.extend(response.output)   # append LLM response to memory

    for item in response.output:
        if item.type == "function_call":
            print("function_call:", item.name, item.arguments)
            call_output = make_call(item)
            messages.append(call_output)    # append tool result to memory
            has_function_calls = True

        elif item.type == "message":
            print("ASSISTANT:")
            print(item.content[0].text)

    it += 1
    if has_function_calls == False:     # exit condition
        break
```

**Why `has_function_calls` resets every iteration:** it tracks whether *this* iteration had tool calls. If it stayed `True` from a previous round, the loop would never stop.

**Exit condition:** no tool calls this round = the LLM gave a final answer = stop.

### What Happens Inside the Loop

```
Iteration 1: LLM calls search("join course") → loop continues
             (tool result appended to memory, LLM hasn't seen it yet)

Iteration 2: LLM calls search("enrollment deadline") → loop continues

Iteration 3: LLM returns plain text message, no tool calls
             → has_function_calls = False → break
```

The loop only stops when the LLM is satisfied and returns a **message** with no tool calls.

### Self-Correction (Typo Example)

```python
agent_loop(instructions, "How do I run Olama locally?")
```

- Iteration 1: LLM searches `"Olama"` → poor results. Also searches `"Ollama"` → finds the answer.
- Iteration 2: LLM gives final answer.

The loop lets the model recover from a bad search **on its own**. No special-case code needed.

### Wrapping in a Function

```python
def agent_loop(instructions, question, model="gpt-4o-mini") -> str:
    messages = [
        {"role": "developer", "content": instructions},
        {"role": "user", "content": question}
    ]
    it = 1

    while True:
        print(f"iteration #{it}...")
        has_function_calls = False

        response = openai_client.responses.create(
            model=model,
            input=messages,
            tools=[search_tool]
        )

        messages.extend(response.output)

        for item in response.output:
            if item.type == "function_call":
                call_output = make_call(item)
                messages.append(call_output)
                has_function_calls = True
            elif item.type == "message":
                last_answer = item.content[0].text
                print(item.content[0].text)

        it += 1
        if has_function_calls == False:
            break

    return last_answer
```

### Steering the Agent with Instructions

**Problem:** the model often stops after one search even when more would help.
**Fix:** update the instructions to push it to search multiple times.

```python
# Before
"Make multiple searches."

# After
"Make multiple searches. First perform search, analyze the results
and then perform more searches."
```

Instructions are guidance, not hard constraints. The LLM may still skip extra searches sometimes.

### Restricting Off-Topic Questions

Without restrictions, the agent answers anything:

```python
agent_loop(instructions, "what's queen gambit?")
# → explains chess opening
```

Scope it with instructions:

```python
"""
The question has to be about the course or its logistics.
Off-topic questions shouldn't be answered.
If the search returns nothing, it's likely an off-topic question.
Only use facts from the FAQ database.
"""
```

Now it says "I couldn't find anything about this in the course FAQ." This is a lightweight **input guardrail** via the prompt.

---

## Key Concepts Summary

| Concept | One-line summary |
|---|---|
| RAG | Search for relevant docs, inject into prompt, let LLM answer |
| minsearch | In-memory keyword search; fast but not persistent |
| sqlitesearch | Disk-based keyword search; persists between sessions |
| Boosting | Give higher weight to matches in important fields (e.g. question) |
| Function calling | LLM requests a tool call; you run it; you send results back |
| Tool definition | JSON schema telling the LLM what functions exist and what args they take |
| Agentic loop | Keep calling LLM + running tools until no tool calls come back |
| Memory | The `messages` list — append everything, pass it every iteration |
| `has_function_calls` | Flag that resets each iteration; `False` = LLM is done |
| Instructions | Developer prompt — the primary way to control agent behavior/scope |
| Self-correction | Loop lets the LLM retry bad searches with different keywords |

---

## The Evolution Across Lessons

```
01  Basic prompt → manual context → RAG function (no search, hardcoded context)
02  minsearch index → search() + build_prompt() + llm() → RAGBase class
03  Switch to SQLite (persistent index on disk)
04  RAGBase + SQLite index (same pattern, swappable index)
05  Function calling — LLM decides when/what to search (1-2 manual API calls)
06  Agentic loop — wrap in while loop, exit when no tool calls
```
