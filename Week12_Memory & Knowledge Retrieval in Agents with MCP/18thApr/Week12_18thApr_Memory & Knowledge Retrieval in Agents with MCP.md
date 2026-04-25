# Week 12 — Memory & Knowledge Retrieval in Agents with MCP (Day 1)
**Date:** April 18, 2026 | **Presenter:** Abhinav Singh (Ramakrishna T)
**Session Duration:** ~2.5 hours

---

## Session Overview
- Why agents forget everything and why that's a problem
- Human memory types mapped to agent memory architecture
- Three types of agent memory: Episodic, Semantic, Procedural
- LangChain memory classes: Buffer, Summary, Window — live demo + test
- Short-term vs Long-term memory — when to use what
- Token cost comparison across memory types (real-world math)
- Vector stores and embeddings — the core concept behind RAG
- FAISS: how Facebook's vector index works under the hood
- Cosine similarity and distance-based retrieval
- Live code demo: semantic search with FAISS + OpenAI embeddings

---

## 1. Why Agents Need Persistent Memory

### The Cold Start Problem

Every time you call `crew.kickoff()` or start a LangChain run, the agent starts from **zero**. No memory of what worked, what failed, what was discussed. This is called a **cold start**.

> **Instructor analogy:** Imagine your best employee — brilliant, solves hard problems. But every morning they wake up with total amnesia. They don't remember yesterday's clients, yesterday's mistakes, or yesterday's breakthroughs. Would you hire them? No. That's exactly what your crew agent does right now.

**Without memory, agents:**
- Repeat the same mistakes every run
- Cannot reference prior context
- Ask the same questions again and again (e.g., asking a customer for their account number every time they contact support)
- Have no accumulated wisdom

**With memory, agents:**
- Recall prior conversations automatically
- Learn from past failures
- Reduce redundant questions
- Improve over time

> **Key Insight:** The difference between support systems customers love vs. hate is often just how memory is configured. The technology is identical — only the memory design differs.

---

## 2. Human Memory → Agent Memory (The Mapping)

### Human Brain Has 3 Memory Systems

| Human Memory | Description | Example |
|---|---|---|
| **Sensory Memory** | Lasts less than a second | A word flashing on screen |
| **Working Memory** | Short-term, limited | Holding a phone number while dialing |
| **Long-Term Memory** | Persistent, deep | How to ride a bicycle; your first job |

### Agents Have the Same 3 Layers

| Agent Equivalent | Technical Term | Capacity |
|---|---|---|
| Sensory Memory | Input / token buffer | Raw incoming text |
| Working Memory | **Context Window** | 4,000 – 128,000 tokens (model dependent) |
| Long-Term Memory | **Vector Store / Database** | Unlimited, persistent, searchable |

> The context window (working memory) is what you pay for every API call. The vector store (long-term memory) lives outside the LLM — you only retrieve what you need.

---

## 3. Three Types of Agent Memory (Episodic, Semantic, Procedural)

These are conceptual categories — understanding them helps you choose the right LangChain class.

### Episodic Memory — "What happened in the past"
- Stores full conversation history, turn by turn
- Example: "The customer said X, you replied Y, then they mentioned Z"
- LangChain class: `ConversationBufferMemory`

### Semantic Memory — "What you just know"
- Stores facts and knowledge, not how/when you learned them
- Example: You know the refund policy — you don't remember when you learned it
- LangChain equivalent: **Vector Store / Knowledge Base** (FAISS, Chroma, Pinecone)

### Procedural Memory — "How to do things automatically"
- Steps and processes you execute without thinking
- Example: The steps to process a refund — you just do them
- LangChain equivalent: **Tools + System Prompts**

> **Q (Rajesh):** Is the type of memory based on the business case?
>
> **A (Abhinav):** Yes, it depends entirely on what you're building. If you don't need to call policy documents for every answer, you don't need to configure semantic memory. If you need a knowledge base, you do. It's a technical architecture decision, not a generic rule.

---

## 4. LangChain Memory Classes — Buffer, Summary, Window

### Setup Code

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import (
    ConversationBufferMemory,
    ConversationSummaryMemory,
    ConversationBufferWindowMemory,
)
from langchain.chains import ConversationChain
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()  # text-embedding-3-small, 1536 dims

# Create 3 memory types
buffer_mem  = ConversationBufferMemory()
summary_mem = ConversationSummaryMemory(llm=llm)  # needs LLM to summarize
window_mem  = ConversationBufferWindowMemory(k=3)  # remember last 3 turns only

# Wrap in chains
chain_buffer  = ConversationChain(llm=llm, memory=buffer_mem,  verbose=False)
chain_summary = ConversationChain(llm=llm, memory=summary_mem, verbose=False)
chain_window  = ConversationChain(llm=llm, memory=window_mem,  verbose=False)
```

### How Each Memory Works

**ConversationBufferMemory (Buffer)**
- Stores **every single turn** in full
- Token cost grows linearly — 100 turns = 100× the memory
- Best for: Very short conversations
- Risk: Token cost blowup; also introduces noise if conversation is very long
  - Example: Customer mentions $29 charge in month 1 and $30 charge in month 6 — buffer passes both, LLM gets confused

**ConversationSummaryMemory (Summary)**
- Compresses all history into a running paragraph
- Uses an LLM internally to summarize (that's why it needs `llm=llm`)
- Always fixed output length — no matter how many turns, output is one compressed paragraph
- Cheaper than buffer, but **lossy** — specific details (like account numbers) can get dropped
- Best for: Long conversations where you need gist, not exact details

**ConversationBufferWindowMemory (Window)**
- Keeps only the last **k** turns (configurable parameter)
- `k=3` → at turn 10, it remembers turns 10, 9, 8 only
- Token cost is approximately fixed (bounded by k × avg turn length)
- Best for: Production systems — instructor's personal recommendation

> **Key Insight (Abhinav's production experience):** "I never used buffer in production because our conversation volumes are very high. Window memory works best for us — users almost always ask from recent context, not something from 5 months ago. We set k=15 or 20 and it works well."

### Live Demo: 10-Turn Customer Conversation

```python
# 10-turn conversation from a single customer
turns = [
    "Hi, my name is Sarah and my account number is AC-7829.",
    "I was charged twice for my subscription last month.",
    "The charge was $29.99 on March 5th and again on March 7th.",
    "I have been a customer for 3 years and this never happened before.",
    "Can you process a refund for the duplicate charge?",
    "How long will the refund take to appear?",
    "Will I get a confirmation email when it is processed?",
    "Also, can you make sure this does not happen next month?",
    "One more thing — can you update my email to sarah.new@email.com?",
    "Thanks. Can you summarise what we discussed today?",
]

for turn in turns:
    chain_buffer.predict(input=turn)
    chain_summary.predict(input=turn)
    chain_window.predict(input=turn)
```

### Recall Test — Results

After 10 turns, ask each chain 3 questions about different turns:

| Test | Buffer | Summary | Window(k=3) |
|---|---|---|---|
| Account number (Turn 1) | ✓ | ✗ | ✗ |
| Charge amount $29.99 (Turn 3) | ✓ | ✓ | ✗ |
| Email update (Turn 9) | ✓ | ✓ | ✓ |

**Why these results:**
- **Buffer** remembers everything — that's its job
- **Summary** compressed the early conversation — account number was a detail that got lost in the paragraph
- **Window(k=3)** at turn 10 only sees turns 10, 9, 8 — turn 1 and turn 3 are completely gone

> **Q (Shobhit):** Why did Summary also not remember the account number? It was supposed to keep a paragraph?
>
> **A (Abhinav):** Summary compresses. When you summarize 10 turns, specific identifiers like account numbers often get dropped — the model keeps the gist of what was discussed, not all specific values. You can fix this with a prompt template that explicitly tells the summarizer: "Always preserve: customer name, account number, order numbers, dollar amounts, dates."

```python
# Fix: Use PromptTemplate to force summary to preserve key fields
from langchain.prompts import PromptTemplate

summary_prompt = PromptTemplate.from_template("""
Progressively summarize the conversation below.
ALWAYS preserve exactly: customer name, account number, order numbers,
dollar amounts, and dates. Never omit these.

Current summary: {summary}
New lines: {new_lines}
New summary:""")

# Pass this to ConversationSummaryMemory as the prompt argument
```

> **Q (Kaviya):** Can we combine window + summary — take only k turns but summarize them?
>
> **A (Abhinav):** You can, but LangChain doesn't support this directly. You'd need to pipe the window output manually to a summary memory. However, in practice, 5 conversation turns is only ~500-5,000 tokens — that's small enough that summarizing them further doesn't save much. The better fix is usually just increasing k.

> **Q (Rafi):** Where is the memory actually stored? RAM or database?
>
> **A (Abhinav):** These are **in-memory** solutions — they live in RAM for the duration of the session. When the process ends, they're gone. You can persist them to a database (like we did with SQLite before), but by default, LangChain memory classes are session-scoped only. Today we'll learn how to store as vectors, which is a persistent solution.

---

## 5. Short-Term vs Long-Term Memory

| | Short-Term (Context Window) | Long-Term (Vector Store) |
|---|---|---|
| **Where** | Inside the LLM prompt | Outside the LLM, in a database |
| **Size** | 4K – 128K tokens (hard limit) | Unlimited (FAISS holds millions of vectors) |
| **Cost** | You pay for every token, every call | Stored once; retrieve only when needed |
| **Speed** | Instant (already in prompt) | Slower: embed → search → retrieve (~2–4 seconds) |
| **Persistence** | Gone after API call | Survives restarts, deployments, updates |
| **Pattern** | Buffer / Summary / Window | RAG (Retrieval Augmented Generation) |

> **The production pattern:** Store everything in long-term (vector store). When a new query arrives, retrieve only the relevant 3–5 documents and inject them into the prompt. This is called **RAG**. The user doesn't see the retrieval — they just get a faster, cheaper, more accurate answer.

### Why ChatGPT Felt Different Before vs. After 2024

Before ChatGPT Plus (pre-2024):
- New chat → blank slate, knew nothing about you
- Long conversation → starts forgetting early messages (context window hit)
- New chat again → total amnesia

After ChatGPT Plus added memory:
- Stores key facts in a **persistent database outside the context window**
- That IS exactly what you're building today with FAISS

> **Instructor:** "Before 2024, every person here probably experienced ChatGPT saying your document is too long, or a new chat having no idea who you are. That was the context window limit. Now they added a vector-based memory feature — the same thing we're building."

---

## 6. Token Cost Comparison — Real-World Math

### The Cost Equation

For the same 15-turn customer support conversation:

| Memory Type | Tokens Used | Cost/Call | Cost/Month (1K daily calls) |
|---|---|---|---|
| Buffer | ~4,000 | $0.00060 | $18.00 |
| Summary | ~800 | $0.00012 | $3.60 |
| Window (k=5) | ~600 | $0.00009 | $2.70 |
| RAG | ~3,000 (only relevant) | $0.00045 | $13.50 |

> **Instructor's real example:** "I built a patient classification system — just intent classification, not even complex reasoning. 300,000–400,000 messages per day. That's ~7 million messages per month. We pay $3,500–$4,000/month just for that one problem, using GPT-4O Mini. One problem. Imagine how many problems a system has."

> **Instructor on Claude costs:** "We got 100 enterprise licenses for Claude. Averaging $2,000/day. That's $40,000/month for 100 users. These companies are making serious money — and spending serious money on training. It's not free anymore."

### Production Recommendation

```
Buffer  → Never use at scale (cost blows up; also adds noise)
Summary → Good if business use case changes slowly (stable policies)
Window  → Best for production (bounded cost; covers recent context)
RAG     → Best for knowledge-heavy queries (retrieve only what's needed)
```

> **Key Insight:** "Choose the wrong memory type and you burn budget on tokens the agent never needed. The finance team doesn't care about your vector embeddings — they care that the API bill went from $27,000/year to $1,600/year."

### Q&A: Why Does RAG Reduce LLM Cost?

> **Q (Sanjay):** To retrieve top 5 similar documents, FAISS must still check every document — doesn't that cost something?
>
> **A (Abhinav):** Great question. The search step is **not an LLM operation** — it's pure matrix multiplication (cosine similarity). You run a mini LLM once to convert your documents to vectors (a one-time activity, maybe monthly, costs < $100 for millions of docs). At query time, you convert the user's query to a vector (mini LLM, nearly free), then compute dot products against your stored vectors (pure CPU/GPU math, no OpenAI call). Only the final step — passing the 3–5 retrieved sentences + query to GPT — costs OpenAI tokens. That's why RAG uses far fewer tokens.

> **Q (Sanjay):** So the embedding model is essentially free?
>
> **A (Abhinav):** Yes. You can use Hugging Face models that are local and free. OpenAI also offers cheap embedding models (text-embedding-3-small is much cheaper than GPT-4). You only pay OpenAI for the final LLM reasoning step.

> **Q (Selvaraju):** If I send the same query twice, does cost increase?
>
> **A (Abhinav):** OpenAI maintains a **cache**. If the same input/context is sent again, they don't re-process it — they return the cached output at a reduced cache cost. Memory and caching are different: memory is on your side (what you include in the prompt), caching is OpenAI's optimization on their side.

---

## 7. Vector Stores and Embedding Fundamentals

### What Is an Embedding?

An embedding is a **translation** — converting words into numbers that **capture meaning**.

> **Key statement (Abhinav):** "Embedding is just a translation. Words → numbers. But not random numbers — numbers that capture meaning."

**Example:**
- "The patient has chest pain" → `[0.23, -0.87, 0.14, 0.56, ..., 1536 numbers]`
- "Client reported cardiac discomfort" → `[0.21, -0.85, 0.12, 0.54, ..., 1536 numbers]`

Different words. Same meaning. **Very similar numbers.** That's the entire concept.

Meanwhile: "The weather is sunny today" → very different numbers (distant in vector space).

### How Similarity Is Measured: Cosine Similarity

- Ranges from **0 to 1**
- Close to 1 = very similar meaning
- Close to 0 = very different meaning
- Identical text = exactly 1

In FAISS, distance metric works inversely — lower distance = more similar.

> **Instructor on dimensions:** "If you have only 2 dimensions (X1, X2), all your values are between 0 and 1 — you can't separate a million documents precisely. More dimensions = more room to spread vectors apart. Similar meanings end up nearby; different meanings end up far apart. text-embedding-3-small gives 1,536 dimensions. The large model gives more. Hugging Face mini models give ~284 or 236 — cheaper but less precise."

### Real-World Analogy: Spotify Recommendation

> Spotify embeds every song into a vector (tempo, key, energy, mood → numbers). Your listening history is also a cluster of vectors. When a new song is added, Spotify calculates: is this new song's vector near your cluster? If yes → recommend it. If not → don't. Your agents use the exact same algorithm for knowledge retrieval.

### Embeddings in Practice — No Keyword Matching

> **Instructor:** "There is no keyword matching here. No word in 'patient has chest pain' matches any word in 'client reported cardiac discomfort.' But their vector similarity is ~0.9. That's the power of semantic search."

---

## 8. Building FAISS Vector Store

### What Is FAISS?

- **Facebook AI Similarity Search** — open-source vector index library
- Instead of checking every document for every query (O(n) search), FAISS clusters vectors during storage
- At query time: find nearest cluster centroid → search only within that cluster
- Result: searches millions in milliseconds, not seconds

> **Instructor:** "This is a data structures problem, not an AI problem. It's software engineering — how do you search n items efficiently? Order of n, n², log n? FAISS came up with their own clustering mechanism. ChromaDB, Pinecone — they each have different mechanisms. LangChain supports all of them."

### Code: Building the Vector Store

```python
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()  # text-embedding-3-small, 1536 dims

# Your "knowledge base" — in production, this would be your policy documents
knowledge_base = [
    "Refund policy: Full refund within 30 days. Partial refund (50%) within 60 days.",
    "Shipping: Standard 3-5 days. Express 1-2 days for $9.99.",
    "Premium members get free express shipping on orders over $50.",
    "Password reset: Settings > Security > Reset Password. Verification email sent.",
    "Account deletion: Contact support. Data removed within 30 days (GDPR).",
    # ... (20 total documents in the demo)
]

# One call — converts all text to vectors and indexes them
store = FAISS.from_texts(knowledge_base, embeddings)

print(f"FAISS store: {store.index.ntotal} documents indexed, {store.index.d} dimensions")
# Output: FAISS store: 20 documents indexed, 1536 dimensions
```

### Code: Semantic Search

```python
# Ask a question in natural language
results = store.similarity_search_with_score("Can I get my money back?", k=1)

for doc, score in results:
    print(f"Score: {score:.3f} | {doc.page_content[:80]}")

# Score is DISTANCE (lower = more similar in FAISS)
# Output example: Score: 0.382 | Gift returns: Recipients get store credit...
```

**Test: Does it find the right document even with different phrasing?**

```python
tests = [
    ("Can I get my money back?",       "Refund"),
    ("How do I change my password?",   "Password"),
    ("Do you deliver overseas?",       "International"),
    ("My item arrived broken",         "Damaged"),
    ("I want to stop paying",          "Cancel"),
]

# All 5 passed — semantic search finds correct documents
# without any keyword matching
```

**Bonus: Cross-Language Test**

```python
for lang, q in [
    ("English", "Can I return a gift?"),
    ("Spanish", "¿Puedo devolver un regalo?"),
    ("French",  "Puis-je retourner un cadeau?"),
    ("Hindi",   "क्या मैं उपहार वापस कर सकता हूं?"),
    ("German",  "Kann ich ein Geschenk zurückgeben?"),
]:
    results = store.similarity_search_with_score(q, k=1)
    print(f"[{lang}] → {results[0][0].page_content[:50]}")

# All 5 languages find the "Gift returns" document
# Embeddings encode MEANING across languages
```

> **Key Insight:** Embeddings work across languages because they encode meaning, not words. The same concept in different languages maps to nearby vectors.

### Live Demo Walkthrough (What Abhinav Did)

1. Created `knowledge_base` list with 20 policy documents
2. Called `FAISS.from_texts(knowledge_base, embeddings)` — this:
   - Sends each document to OpenAI text-embedding-3-small
   - Gets back 1,536-dimensional vectors
   - Stores (text, vector) pairs in the FAISS index
3. Called `store.similarity_search_with_score("How long does delivery take?", k=2)`
4. FAISS returned: "Shipping: Standard 3-5 days..." (score 0.35) and "Premium members get free express shipping..." (score 0.41)
5. Explained: "See — the query asked about delivery, not 'shipping'. No keyword match. But the meaning matched."

> **Q (Gaurav):** What is `FAISS.from_texts`? Is that the function name?
>
> **A (Abhinav):** Yes. `OpenAIEmbeddings` gives you the numbers (the mini LLM). `FAISS` is the module that stores and retrieves those numbers efficiently. They do different things.

> **Q (Rafi):** If I want to remove old data from the vector store, can I?
>
> **A (Abhinav):** The simplest approach is to **rebuild** — run the same script with only the documents you want. You can also add metadata (like month number) and filter by it at query time. For a use case where you want only the last 3 months of data, schedule a rebuild script to run monthly and only pass recent documents.

---

## 9. Distance vs Cosine Similarity

| Metric | Range | Interpretation | FAISS default |
|---|---|---|---|
| Cosine Similarity | 0 to 1 | Higher = more similar | No (but available) |
| L2 Distance | 0 to ∞ | Lower = more similar | Yes (default) |

In the demo output, `score` values like `0.38` are L2 distances (lower = closer = more similar).

> **Instructor:** "If the score is less than 0.5, you can design your solution to say: this document is not similar enough — don't include it in the prompt. This prevents the LLM from hallucinating based on irrelevant context."

```python
# Design pattern: filter by similarity threshold
SIMILARITY_THRESHOLD = 0.5

results = store.similarity_search_with_score(query, k=3)
relevant_docs = [doc for doc, score in results if score < SIMILARITY_THRESHOLD]

if not relevant_docs:
    answer = "I don't have information about that in my knowledge base."
else:
    # Pass relevant_docs to LLM
    ...
```

---

## 10. Industry Context — Claude and LLMs in the Wild

During the session, an extended discussion happened about Claude's capabilities:

**Kiran Kumar (student):** "We implemented a GenAI SDLC system using Claude — fully streamlining grooming and code generation. It went from POC to live last week."

**Abdul (student):** "We're using RooCode (VS Code plugin) with Claude Sonnet as the backend."

**Abhinav's assessment of Claude Opus 4.6/4.7:**
- "Claude is the best in the market right now, in my experience"
- "I gave it an architecture — 10 engineers built over 30 days — Claude rebuilt it in 4-5 hours, more optimized, lower latency"
- "Claude Design was released yesterday — caused a 5-10% drop in Figma's stock market value"
- "Claude can generate 20,000+ lines of code. You can't review it line by line. The skill now is **testing** what it generates, not reading it"
- **Key advice:** Generate modules, test them. Testing is the critical skill in the AI era.

---

## Key Takeaways

| Concept | Summary |
|---|---|
| Cold Start Problem | Agents forget everything between runs — memory is required for real utility |
| 3 Memory Types | Episodic (history) → Buffer; Semantic (knowledge) → Vector Store; Procedural → Tools + Prompts |
| Buffer Memory | Stores all turns; most accurate but most expensive; never use at scale |
| Summary Memory | Compresses to paragraph; loses specific details; use prompts to force key fields |
| Window Memory | Keeps last k turns; approximately fixed cost; recommended for production |
| Short-Term | Context window (4K–128K tokens); fast, expensive, volatile |
| Long-Term | Vector store; persistent, cheap, ~2–4 sec retrieval latency |
| RAG | Retrieve top 3–5 relevant docs by meaning, inject into prompt; reduces tokens and hallucination |
| Embeddings | Text → 1,536-dimensional numbers capturing meaning; similar meanings = nearby vectors |
| FAISS | Facebook's clustering-based vector index; fast retrieval without checking every document |
| Cosine Similarity | 0–1; higher = more similar meaning; works across languages |
| Cost Trade-off | Buffer >> Summary > Window ≈ RAG; choose based on business accuracy requirements |

---

## Files in This Folder

| File | Purpose |
|---|---|
| `Week12_18thApr_Memory & Knowledge Retrieval in Agents with MCP.md` | Session notes (this file) |
| `0 - 74408 - Live session on Memory and Knowledge Retrieval in Agents with Ramkrishna - TRANSCRIPT.txt` | Full WEBVTT transcript |
| `16675 - Ramkrishna - 18th March 2026.zip` | Raw session zip |
| `18thApr_extracted/Week12_Teaching_Notebook.ipynb` | Complete teaching notebook (8 parts with tests) |
| `18thApr_extracted/Week12_Student.pdf` | Student reference PDF |
