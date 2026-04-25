# Week 12 — Memory & Knowledge Retrieval in Agents with MCP (Day 2)
**Date:** April 19, 2026 | **Presenter:** Ramakrishna T

---

## Session Overview
- Day 1 recap: memory types, cosine similarity, ConversationWindowMemory is cheapest
- RAG pipeline: Retrieve → Augment → Generate using `RetrievalQA.from_chain_type()`
- Hallucination test: RAG vs. bare LLM on the same question
- PDF document ingestion: 5-step pipeline using PyPDFLoader + RecursiveCharacterTextSplitter + FAISS
- Chunking theory: why overlap prevents context loss at chunk boundaries
- Memory loop: storing each Q&A interaction back into FAISS so the agent improves over time
- Multi-store RAG: VDB1 (knowledge) + VDB2 (thumbs up) + VDB3 (thumbs down) — RLHF at inference time
- FAISS persistence: `save_local()` and `load_local()` so you don't rebuild on every startup
- Model Context Protocol (MCP): what it is, 3-layer architecture, how it differs from RAG
- MCP multi-tool demo: PatientLookup (RAG) + DrugInteractionCheck (rule-based) + DoseCalculator (math)
- Conversational agent capstone: ClinicalAssistant combining PDF knowledge + conversation memory
- Group exercise: participants build domain-specific RAG agents

---

## 1. Day 1 Recap

Before new content, Ramakrishna reviewed the three key takeaways from Day 1:

| Concept | Recap |
|---------|-------|
| **Memory types** | Episodic = Buffer, Semantic = Vector Store, Procedural = Tools/System Prompt |
| **Cost order** | Buffer > Summary > Window ≈ RAG; Window + RAG is usually the sweet spot |
| **Cosine similarity** | Measures angle between embedding vectors; FAISS uses L2 by default (lower = closer) |

> **Key Insight:** "Window memory is cheapest not because it's smart, but because it throws away old context. RAG is cheapest in the long run because it retrieves only what's relevant."

---

## 2. RAG Pipeline — RetrievalQA in Practice

### What RAG Does

```
  ┌──────────────────────────────────────────────┐
  │               User Question                  │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼  RETRIEVE
  ┌──────────────────────────────────────────────┐
  │   FAISS Semantic Search                      │
  │   query → 1536-dim vector → top-k chunks     │
  └──────────────────────┬───────────────────────┘
                         │  top-k chunks
                         ▼  AUGMENT
  ┌──────────────────────────────────────────────┐
  │   Build Prompt                               │
  │   "Context: {chunks}\n\nQuestion: {query}"   │
  └──────────────────────┬───────────────────────┘
                         │
                         ▼  GENERATE
  ┌──────────────────────────────────────────────┐
  │   LLM  (gpt-4o-mini)                         │
  │   Grounded answer  +  source citations       │
  └──────────────────────────────────────────────┘
```

### Setup — Knowledge Base + FAISS Store

```python
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv
load_dotenv()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings()  # text-embedding-3-small, 1536 dims

# 20 customer support policy documents
knowledge_base = [
    "Refund policy: Full refund within 30 days. Partial refund (50%) within 60 days.",
    "Shipping: Standard 3-5 days. Express 1-2 days for $9.99.",
    "Premium members get free express shipping on orders over $50.",
    "Password reset: Settings > Security > Reset Password.",
    "Account deletion: Contact support. Data removed within 30 days (GDPR).",
    "Payment: Visa, MasterCard, Amex, PayPal, Apple Pay, Google Pay.",
    "Order tracking: Tracking link emailed after shipping. 24 hours to activate.",
    "International shipping: 45 countries. 7-14 days. Import duties customer responsibility.",
    "Gift returns: Recipients get store credit. Original purchaser gets refund.",
    "Damaged items: Report within 48 hours with photos. Free replacement in 2 days.",
    "Subscriptions: Basic $9.99/mo, Pro $19.99/mo, Enterprise custom.",
    "Cancellation: Cancel anytime, no fee. Refund for unused portion.",
    "Price matching: Match competitor prices within 14 days with proof.",
    "Bulk orders: 50+ units = 15% discount. Contact sales@company.com.",
    "Warranty: Electronics 2-year warranty. Extended warranty $29.99/year.",
    "Store hours: Online 24/7. Support Mon-Fri 8am-8pm EST.",
    "Loyalty: 1 point per dollar. 100 points = $5 discount. Expire after 12 months.",
    "Size exchanges: Free within 30 days. Unworn with tags.",
    "Two-factor auth: Settings > Security > 2FA.",
    "Data privacy: GDPR/CCPA compliant. Full data export available anytime.",
]

store = FAISS.from_texts(knowledge_base, embeddings)
# store.index.ntotal = 20
```

### RetrievalQA Chain

```python
from langchain.chains import RetrievalQA

retriever = store.as_retriever(search_kwargs={"k": 3})  # fetch top 3 similar chunks
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",           # "stuff" = inject all retrieved docs as-is into prompt
    retriever=retriever,
    return_source_documents=True, # also return which docs were used
)

result = rag_chain.invoke({"query": "I bought something 45 days ago. Can I still get a refund?"})
print(result['result'])           # grounded LLM answer
print(result['source_documents']) # list of Document objects that were retrieved
```

**`chain_type` options:**
| Type | Behavior | Best for |
|------|----------|----------|
| `"stuff"` | All docs injected directly | ≤ 5 short chunks |
| `"map_reduce"` | LLM summarizes each doc separately, then combines | Many / long docs |
| `"refine"` | Iteratively refines answer as each doc is added | Complex reasoning |

### Hallucination Test — RAG vs. Bare LLM

```python
test_q = "What is your refund policy for items returned after 30 days?"

rag_result  = rag_chain.invoke({"query": test_q})   # grounded
bare_result = llm.invoke([HumanMessage(content=test_q)]).content  # no grounding

# RAG says: "50% partial refund within 60 days" (cites the stored policy)
# Bare LLM may confidently say: "No refunds after 30 days" — WRONG
```

> **Key Insight:** "Bare LLM is like a student who studied hard but read wrong books. It sounds confident but gives wrong company-specific answers. RAG points it at the right book."

### Q&A — Hallucination Test

**Ravi:** If the retrieved chunks already contain the answer, why do we need the LLM? Can we just return the chunk directly?

**Ramakrishna:** You can, but the chunk is a raw policy sentence. The LLM adds reasoning — it interprets "within 60 days" given the user's "I bought something 45 days ago" context and says "yes, 50% partial refund applies." The chunk answers WHAT; the LLM answers WHAT DOES THIS MEAN FOR YOU.

---

## 3. PDF Document Ingestion — 5-Step Pipeline

The instructor demonstrated loading 5 patient PDF files and querying across all of them.

```
  ┌──────────────────────────────────────────────┐
  │   PDF Files  (patient_*.pdf × 5)             │
  └──────────────────────┬───────────────────────┘
                         │  Step 1 · Load
                         ▼  PyPDFLoader
  ┌──────────────────────────────────────────────┐
  │   Documents  (1 object per page, raw text)   │
  └──────────────────────┬───────────────────────┘
                         │  Step 2 · Chunk
                         ▼  RecursiveCharacterTextSplitter
  ┌──────────────────────────────────────────────┐
  │   Chunks  (800 chars, overlap = 100)         │
  └──────────────────────┬───────────────────────┘
                         │  Step 3 · Embed
                         ▼  OpenAIEmbeddings
  ┌──────────────────────────────────────────────┐
  │   Vectors  (1536-dim float array per chunk)  │
  └──────────────────────┬───────────────────────┘
                         │  Step 4 · Index
                         ▼  FAISS.from_documents()
  ┌──────────────────────────────────────────────┐
  │   FAISS Vector Store  (searchable by meaning)│
  └──────────────────────┬───────────────────────┘
                         │  Step 5 · Query
                         ▼  RetrievalQA
  ┌──────────────────────────────────────────────┐
  │   Grounded Answer  +  Source Citations       │
  └──────────────────────────────────────────────┘
```

### Step 1 — Load PDFs

```python
from langchain_community.document_loaders import PyPDFLoader
import glob, os

pdf_files = sorted(glob.glob("test_patients/*.pdf"))
all_docs = []
for path in pdf_files:
    loader = PyPDFLoader(path)
    docs = loader.load()         # returns list of Document objects (one per page)
    all_docs.extend(docs)
    print(f"Loaded: {os.path.basename(path)} ({len(docs)} pages)")

# Output:
# Loaded: patient_1_pediatric.pdf (1 pages, 680 chars)
# Loaded: patient_2_pregnancy.pdf (1 pages, 720 chars)
# ...
# Total pages loaded: 5
```

> **Note on scanned PDFs:** PyPDFLoader only works on text-layer PDFs. For scanned images, use `UnstructuredPDFLoader` — it runs OCR first.

### Step 2 — Chunk Documents

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,       # max characters per chunk
    chunk_overlap=100,    # overlap between consecutive chunks
    separators=["\n\n", "\n", ". ", " "]  # tries these separators in order
)

chunks = splitter.split_documents(all_docs)
print(f"Total chunks: {len(chunks)}")  # 5 short PDFs → 5-15 chunks
```

**Why overlap matters:**

```
Document:  "...Patient has hypertension. Creatinine is 1.8. He is currently..."
                     ↑ chunk 1 ends here       ↑ chunk 2 starts here

Without overlap:  chunk 1 has "hypertension", chunk 2 has "Creatinine"
                  → a query about "creatinine in hypertension patients" retrieves neither cleanly

With overlap=100: chunk 2 starts 100 chars earlier
                  → both chunks contain the link between hypertension and creatinine
```

> **Key Insight:** "Overlap is like the margin in a book page. When you cut pages, you keep a little of the previous page so you don't lose the sentence that was split in the middle."

**Recommended settings:** chunk_size=800, overlap=100 for most documents (500–1000 range is safe).

### Steps 3-4 — Embed + Index

```python
pdf_store = FAISS.from_documents(chunks, embeddings)
print(f"Chunks indexed: {pdf_store.index.ntotal}")  # e.g., 15
print(f"Dimensions:     {pdf_store.index.d}")        # 1536
```

### Step 5 — Query with Citations

```python
pdf_retriever = pdf_store.as_retriever(search_kwargs={"k": 4})
pdf_rag = RetrievalQA.from_chain_type(
    llm=llm, chain_type="stuff",
    retriever=pdf_retriever,
    return_source_documents=True,
)

questions = [
    "What medications is patient 5 currently taking?",
    "Which patient has renal issues and what is their eGFR?",
    "Is there a pregnant patient? What NSAID concern exists?",
    "What drug interactions should we be concerned about for the cardiac patient?",
    "Which patient across all 5 files is at the HIGHEST risk and why?",
]

for q in questions:
    result = pdf_rag.invoke({"query": q})
    print(f"Q: {q}")
    print(f"A: {result['result'][:250]}")
```

**Expected answers:**
- Q1: Lists Clarithromycin, Clopidogrel, Atorvastatin (from patient_5_cardiac.pdf)
- Q2: Patient 3 Robert Chen, eGFR 42, CKD Stage 3
- Q3: Patient 2 Sarah Martinez, Naproxen concern in 3rd trimester (premature ductus arteriosus)
- Q4: CYP3A4 interaction — Clarithromycin inhibits Clopidogrel metabolism
- Q5: Cross-document comparison — agent reads ALL 5 PDFs and ranks risk

> **Key Insight:** "Q5 is the 'wow' moment. The agent reads 5 separate PDFs and reasons across them — like a doctor who has reviewed all patient files before the consultation."

---

## 4. Memory Loop — The Agent That Improves With Every Conversation

### Concept

After each interaction, the agent stores its own Q&A back into the FAISS store. On the next query, it retrieves not just knowledge documents but also its own prior conversations — if similar questions were asked before.

```
  New Query
      │
      ▼
  ┌──────────────┐  top-k  ┌──────────────┐  response
  │   RETRIEVE   │ chunks  │   GENERATE   │ ─────────────► (to user)
  │  FAISS search│────────►│     LLM      │
  │ docs + Q&As  │         │  context →   │
  └──────▲───────┘         │  response    │
         │                 └──────┬───────┘
         │                        │
         │                        ▼
         │                ┌──────────────┐
         │                │    STORE     │
         │                │  add_texts() │
         │                │  Q&A record  │
         │                └──────┬───────┘
         │                       │ new vector appended
         └───────────────────────┘
         FAISS grows richer → retrieved on next query
```

### MemoryLoopAgent Class

```python
class MemoryLoopAgent:
    def __init__(self, name, knowledge_texts):
        self.name = name
        self.store = FAISS.from_texts(knowledge_texts, embeddings)
        self.interactions = []
        self.initial_count = self.store.index.ntotal   # track original doc count

    def respond(self, query):
        # RETRIEVE
        retrieved = self.store.similarity_search_with_score(query, k=3)
        context = "\n".join([doc.page_content for doc, _ in retrieved])

        # GENERATE
        response = llm.invoke([
            SystemMessage(content=f"You are {self.name}. Use ONLY this context:\n{context}"),
            HumanMessage(content=query)
        ]).content

        # STORE — this is the memory loop
        record = f"[STORED] Customer asked: {query} | Agent answered: {response[:120]}"
        self.store.add_texts([record])   # ← adds to FAISS on the fly
        self.interactions.append({"query": query, "response": response[:120]})

        return response, retrieved

agent = MemoryLoopAgent("SupportBot", knowledge_base)

queries = [
    ("Q1", "What is the refund policy?"),
    ("Q2", "How do I track my order?"),
    ("Q3", "Can I cancel my subscription?"),
    ("Q4", "What payment methods do you accept?"),
    ("Q5", "Do you price match?"),
    ("Q6", "What did I ask you about refunds earlier?"),  # MEMORY TEST
]

for label, q in queries:
    response, _ = agent.respond(q)
    print(f"{label}: {q}")
    print(f"       {response[:100]}...\n")
```

### Validating the Memory Loop

```python
# After Q1-Q5 are stored, test if Q6 retrieves the stored Q1 interaction
results = agent.store.similarity_search_with_score("What did I ask about refunds?", k=5)
for doc, score in results:
    is_stored = "[STORED]" in doc.page_content
    marker = "← STORED INTERACTION" if is_stored else "← ORIGINAL DOC"
    print(f"[{score:.3f}] {marker}: {doc.page_content[:70]}...")

# Expected: at least one result marked "← STORED INTERACTION"
# store.index.ntotal = 20 (original) + 5 (stored interactions) = 25
print(f"Store size: {agent.store.index.ntotal}")
```

### Q&A — Memory Loop Storage

**Selvaraju:** If we keep storing all conversations, we'll need more and more storage space. Is that a problem?

**Ramakrishna:** Space grows, yes. But you don't have to store everything blindly. You can add filters: store only when user gives thumbs up, or only when the query is about certain categories. `store.add_texts()` is just a call — you control when you make it.

**Abhishek:** Does adding conversation records affect the original chunking mechanism?

**Ramakrishna:** No. Chunking happens once, offline, when you load PDFs. The FAISS index is already built. When you call `add_texts`, it just appends new vectors to the existing index. The original 20 docs stay intact — we just add new vectors alongside them.

**Ravi Pitla:** When Q6 retrieves stored interactions, does it just return them directly, or does it still go through the LLM?

**Ramakrishna:** That's up to your design. What I showed here is just retrieval-only — I'm checking whether the stored interactions appear in the results at all. In production, you'd pass those retrieved interactions as context to the LLM: "Here is a prior conversation where you answered X. Use this to answer the current question."

**Deepak:** Can I print and see exactly what context RAG added to my prompt?

**Ramakrishna:** Yes. When you use `return_source_documents=True`, the `result['source_documents']` list shows exactly what was retrieved. Each document's `.page_content` is what gets injected into the prompt. The `[STORED]` prefix in stored interactions is a design choice — it lets you tell at a glance whether a retrieved result is an original document or a stored conversation.

---

## 5. Multi-Store RAG — Thumbs Up + Thumbs Down Architecture

This was the most advanced design concept of the session.

### The Problem with Single-Store Memory

If you store all interactions in one store, you mix good answers with bad ones. The model sees both but can't tell them apart.

### Three-Vector-Store Architecture

```
                              New Query
                                  │
           ┌──────────────────────┼──────────────────────┐
           │                      │                      │
           ▼                      ▼                      ▼
  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
  │  VDB1           │  │  VDB2           │  │  VDB3           │
  │  Original       │  │  Thumbs-Up      │  │  Thumbs-Down    │
  │  Knowledge      │  │  Interactions   │  │  Interactions   │
  │  (policy docs,  │  │  (Q&A where     │  │  (Q&A where     │
  │   FAQs, etc.)   │  │   user happy)   │  │  user unhappy)  │
  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘
           │ top-3 docs         │ top-3 good          │ top-3 bad
           └────────────────────┼─────────────────────┘
                                │
                                ▼
              ┌─────────────────────────────────────────┐
              │             LLM Prompt                  │
              │  query:         <current question>      │
              │  context:       <docs from VDB1>        │
              │  good examples: <Q&A pairs from VDB2>   │
              │  bad examples:  <Q&A pairs from VDB3>   │
              └─────────────────────────────────────────┘
                                │
                                ▼
                       Grounded Answer
                  (less likely to repeat past mistakes)
```

The LLM now receives:
- What the policy says (VDB1)
- What worked well for similar questions (VDB2)
- Where it made mistakes on similar questions (VDB3)

> **Key Insight:** "It's like teaching your kid before an exam. You say: here are the 10 questions you answered correctly before. Here are the 10 similar questions you got wrong. Now here's the exam question. How do you answer it?"

### RLHF Connection

**Abhishek:** Is this the same as training the LLM? It feels like reinforcement learning.

**Ramakrishna:** Exactly right — this pattern is inspired by RLHF (Reinforcement Learning with Human Feedback). But there's no training happening here. We're not updating model weights. We're doing RLHF at **inference time** — injecting the human feedback signal directly into the prompt context. Same idea, zero training cost.

### Saturation Point

As you keep adding thumbs-up and thumbs-down interactions to VDB2 and VDB3, something interesting happens:

- The vector stores grow richer
- When a new query arrives, the retrieved similar questions become more and more precise
- Eventually you have exact matches — "this user asked this exact question before and the answer was X"
- The model stops making the same mistakes. This is the **saturation point**

> "When your database becomes bigger and bigger, you have much more control on similarity, much richer representation. You get closer and closer to exact matches. The model reaches a saturation where it almost never repeats old mistakes."

### Critical Warning — Don't Blindly Trust Customer Ratings

> **Key Insight (Ramakrishna, very emphatic):** "Do NOT assume customers are always right. About 20% of the time, customers rate incorrectly — they click thumbs-up when they meant thumbs-down, or they're wrong about the answer. If you keep adding wrong examples at 20% error rate, your RAG becomes a source of misinformation."

**Recommended validation before storing:**

```python
# Before adding to VDB2 or VDB3, validate with LLM confidence score
validation_prompt = f"""
Query: {query}
Context (ground truth from policies): {policy_context}
Agent answer: {response}
User feedback: Thumbs {'up' if thumbs_up else 'down'}

Judge: Is the user's rating correct?
Give a confidence score 0-100 that the user rating accurately reflects answer quality.
"""

confidence = llm.invoke([...]).content  # Extract numeric score

if confidence >= 95:
    vdb2.add_texts([good_record])     # Store as confirmed good
elif confidence < 60:
    pass                              # Discard — customer likely wrong
else:
    pass                              # 60-95%: borderline, probably skip
```

> "80% of the time customers are right; 20% they're wrong. That's our simulation baseline. A thumbs-up or thumbs-down is a signal, not a fact. Validate before trusting."

---

## 6. FAISS Persistence — Save and Load Vector Stores

### The Problem (Kannan's Question)

**Kannan:** In your demo, you rebuild the entire FAISS store every time you run the notebook — load PDFs, chunk them, embed them, index them. Doesn't that mean you pay the embedding cost and wait 15+ minutes every time the application restarts?

**Ramakrishna:** Exactly right. That's the problem. In production, you run the indexing pipeline ONCE (or on a schedule — monthly, weekly), save the store to disk, then just load the binary on startup.

### Save and Load

```python
# Save the store to disk (creates 2 files: index.faiss and index.pkl)
store.save_local("vector_db")

# Load it back (on next startup — no re-embedding needed)
store = FAISS.load_local(
    "vector_db",
    embeddings,
    allow_dangerous_deserialization=True  # required flag for pickle
)
```

**What gets saved:** Two binary files are created:
- `index.faiss` — the FAISS index (all the vectors, binary format)
- `index.pkl` — the document objects (text + metadata), stored as Python pickle

**Kannan:** Can I view what's inside the .faiss file?

**Ramakrishna:** No — it's binary. You have to load it as a Python object and then inspect programmatically. Use `store.index.ntotal` to count vectors, or `store.similarity_search(...)` to test queries.

**Kannan:** What kind of database is the pickle file?

**Ramakrishna:** It's Python's native object serialization — not a proper database. For production systems with millions of documents, use a dedicated vector database: Pinecone, Weaviate, pgvector in Postgres, or cloud equivalents in AWS Aurora / Azure. FAISS local is fine for prototyping and small-medium datasets.

---

## 7. Model Context Protocol (MCP)

### What Is MCP?

> "Think of HTTP — it's the protocol that lets your browser talk to any website. MCP is the HTTP for AI agents. It's the protocol that lets your agent talk to any tool."

MCP (Model Context Protocol) was developed by Anthropic as a standard for how AI agents connect to the external world.

### MCP vs. RAG

| | RAG | MCP |
|--|-----|-----|
| **What is it** | A pattern (retrieve → augment → generate) | A protocol (like HTTP or TCP/IP) |
| **Scope** | Document retrieval only | Any external capability |
| **Analogy** | One road | The entire highway system |
| **Examples** | Search PDFs, query knowledge base | Search PDFs, query databases, call APIs, run code, read files, post to Slack |

> "RAG is one tool in the MCP toolbox. MCP is the toolbox itself."

### Three-Layer Architecture

```
  ┌──────────────────────────────────────────────────────────┐
  │  HOST  —  Your Application                               │
  │  Python script · ChatGPT · Claude Desktop · VS Code      │
  │                                                          │
  │  ┌─────────────────────────────────────────────────────┐ │
  │  │  CLIENT  —  Connection Manager                      │ │
  │  │  · One client per MCP server                        │ │
  │  │  · Formats requests / parses responses              │ │
  │  │  · Handles authentication                           │ │
  │  │  · Discovers available tools via mcp.listTools()    │ │
  │  └──────────────────────────┬──────────────────────────┘ │
  └─────────────────────────────┼────────────────────────────┘
                                │
                    JSON-RPC  (stdio / HTTP / SSE)
                                │
                                ▼
  ┌──────────────────────────────────────────────────────────┐
  │  MCP SERVER  —  Exposes External Capabilities            │
  │                                                          │
  │  ┌────────────────┐  ┌────────────────┐  ┌────────────┐  │
  │  │    TOOLS       │  │   RESOURCES    │  │  PROMPTS   │  │
  │  │  Functions,    │  │  Files, DB     │  │  Reusable  │  │
  │  │  APIs, calcs,  │  │  records,      │  │  system    │  │
  │  │  web search    │  │  live feeds    │  │  templates │  │
  │  └────────────────┘  └────────────────┘  └────────────┘  │
  │                                                          │
  │  Backends: SQL DBs · REST APIs · File Systems            │
  │            Slack/Teams · Code Sandboxes · Vector Stores  │
  └──────────────────────────────────────────────────────────┘
```

**Three things an MCP Server exposes:**
- **Tools** — callable functions or APIs (database queries, calculations, API calls)
- **Resources** — data sources to read (files, database records, live feeds)
- **Prompts** — reusable system prompts shared across agents (avoids rewriting the same system prompt for every agent in the same domain)

### The Problem MCP Solves

**Without MCP:**
```
  ┌─────────┐ ──────────────────────────────► Tool A  (custom code)
  │ Agent 1 │ ──────────────────────────────► Tool B  (custom code)
  └─────────┘
  ┌─────────┐ ──────────────────────────────► Tool A  (duplicate!)
  │ Agent 2 │ ──────────────────────────────► Tool C  (custom code)
  └─────────┘
  ┌─────────┐ ──────────────────────────────► Tool B  (duplicate!)
  │ Agent 3 │ ──────────────────────────────► Tool D  (custom code)
  └─────────┘
  ...500 agents

  Add Tool E?  → update every agent script.
  Remove Tool A? → update every agent script.
```

**With MCP:**
```
  ┌─────────┐ ─┐
  │ Agent 1 │  │               ┌─────────────────────────┐
  └─────────┘  │               │       MCP SERVER        │──► Tool A
  ┌─────────┐  ├──────────────►│                         │──► Tool B
  │ Agent 2 │  │               │  all tools registered   │──► Tool C
  └─────────┘  │               │  once, served to all    │──► Tool D
  ┌─────────┐  │               │  agents on demand       │──► Tool E
  │ Agent 3 │ ─┘               └─────────────────────────┘
  └─────────┘
  ...500 agents

  Add Tool F?  → update server once. All agents see it immediately.
  Remove Tool A? → update server once. No agent code to touch.
```

> **Key Insight:** "Without MCP, adding a new tool means going into every agent script and adding it manually. With MCP, you add the tool to the server once, update the tool description in your config, and all agents can use it."

### Q&A — MCP Architecture

**Rajesh:** What about redundancy? If you rely on one MCP server, is that a single point of failure?

**Ramakrishna:** Yes, and this is exactly why you should NOT have one massive MCP server for your entire company. The risk is two-fold: (1) single point of failure, and (2) if you put 100 tools in one server, the LLM hallucinates when choosing which tool to call.

The recommendation is **domain-specific MCP servers**:
- Support team → Support MCP Server (customer queries, refunds, order tracking)
- Sales team → Sales MCP Server (pricing, quotes, deals)
- Marketing → Marketing MCP Server (campaign data, analytics)

Within a domain, the LLM makes accurate tool choices. Across domains with 100 mixed tools, it gets confused.

**Rajesh:** Can you restrict which agent can access which tools? Finance tools shouldn't be accessible from sales agents.

**Ramakrishna:** Yes. The client-to-server connection requires authentication. You can also restrict at the tool level — which tools a given client (agent) is allowed to invoke. All configurable.

**Kaviya:** We already use MCP at work — Windsurf is our host, and it connects to GitLab, Splunk, SonarQube, and Jira. When I ask "what Jira tickets are assigned to me in April sprint," it fetches them. What's the client here?

**Ramakrishna:** Your Windsurf agent is the client. It sits inside Windsurf (the host), manages the connection to the Jira server, and when you ask a question, the client identifies the right tool (Jira query) and sends a formatted request to the server. The server returns the tickets, and the client hands that context back to the LLM for synthesis. Client = the piece of code inside your local system that knows how to make a properly formatted request to the server.

### Tool Description Quality Is Critical

> **Key Insight:** "If your tool description is wrong or vague, the LLM will pick the wrong tool. Don't blame MCP for failing — check your descriptions. Good descriptions = good tool selection. If a tool is being called incorrectly, 90% of the time the description is the problem."

**Testing tool routing:**
```python
tool_descriptions = "\n".join([f"- {t.name}: {t.description}" for t in tools])
tool_choice = llm.invoke([
    SystemMessage(content=f"You are a tool router. Given a user query, respond with ONLY the tool name.\nAvailable tools:\n{tool_descriptions}"),
    HumanMessage(content=query)
]).content.strip()
```

### Team Responsibilities with MCP

| Role | Responsibility |
|------|---------------|
| **AI/ML Engineers** | Build agents, write tool descriptions, test tool routing, design prompts |
| **Software Engineers** | Build and maintain the MCP server, implement tool functions, manage API connections |
| **Both together** | Define what tools are needed, validate that descriptions are accurate |

> "Software engineering builds the server and the tools. We (AI/ML) build the agents. We collaborate on what tools exist and how to describe them well. If we describe them poorly, the agents fail — but that's our fault, not the software team's."

**Amit:** With MCP, will we need to make minimum updates to our agents when tools change?

**Ramakrishna:** Exactly. If the tool's internal implementation changes (the server side), the agent doesn't know and doesn't care — it only knows the name and description. As long as the description doesn't change, the agent code doesn't change. That's the whole point.

---

## 8. MCP Multi-Tool Demo — Clinical Assistant

The demo showed 3 tools with completely different implementation types — one RAG (retrieval), one rule-based, one mathematical.

### The Three Tools

```python
from langchain.tools import Tool

# TOOL 1: PatientLookup — RAG (vector search)
def patient_lookup_fn(query: str) -> str:
    '''Search patient PDFs for relevant medical data.'''
    results = pdf_store.similarity_search_with_score(query, k=3)
    parts = []
    for doc, score in results:
        src = os.path.basename(doc.metadata.get('source', 'unknown'))
        parts.append(f"[{src} | score:{score:.3f}] {doc.page_content[:200]}")
    return "\n".join(parts)

# TOOL 2: DrugInteractionCheck — Rule-based lookup (NOT retrieval)
KNOWN_INTERACTIONS = {
    ("clarithromycin", "clopidogrel"): "SEVERE: CYP3A4 inhibition reduces Clopidogrel activation. Risk: cardiac event.",
    ("clarithromycin", "atorvastatin"): "MODERATE: CYP3A4 inhibition increases Atorvastatin levels. Risk: rhabdomyolysis.",
    ("metformin", "ibuprofen"): "MODERATE: Ibuprofen reduces renal clearance of Metformin. Risk: lactic acidosis in CKD.",
    ("warfarin", "aspirin"): "SEVERE: Additive anticoagulation. Risk: GI bleeding.",
    ("naproxen", "pregnancy"): "SEVERE: NSAIDs in 3rd trimester risk premature ductus arteriosus closure.",
}

def drug_interaction_fn(query: str) -> str:
    '''Check for known drug-drug interactions.'''
    query_lower = query.lower()
    found = []
    for (drug_a, drug_b), warning in KNOWN_INTERACTIONS.items():
        if drug_a in query_lower and drug_b in query_lower:
            found.append(f"{drug_a.title()} + {drug_b.title()}: {warning}")
    return "\n".join(found) if found else "No known interactions found."

# TOOL 3: DoseCalculator — Pure math (NOT retrieval)
def dose_calculator_fn(query: str) -> str:
    '''Calculate medication dose based on patient weight in kg.'''
    import re
    weight_match = re.search(r'(\d+)\s*kg', query.lower())
    weight = float(weight_match.group(1)) if weight_match else 70.0

    dose_table = {
        "metformin":    (15,  "mg", "twice daily",      2000),
        "ibuprofen":    (10,  "mg", "every 6-8 hours",   800),
        "amoxicillin":  (25,  "mg", "every 8 hours",    1500),
        "methotrexate": (0.3, "mg", "once weekly",        25),
    }
    for drug, (per_kg, unit, freq, max_dose) in dose_table.items():
        if drug in query.lower():
            calc = min(per_kg * weight, max_dose)
            return f"{drug.title()} for {weight}kg: {calc:.0f} {unit} {freq}"
    return "Drug not found."

# Register all 3 as LangChain Tools
tools = [
    Tool(name="PatientLookup",
         description="Search patient medical records from uploaded PDFs. Use when asked about patient history, symptoms, diagnoses, or current medications.",
         func=patient_lookup_fn),
    Tool(name="DrugInteractionCheck",
         description="Check for known drug-drug interactions. Use when asked if two drugs interact or if a drug combination is safe.",
         func=drug_interaction_fn),
    Tool(name="DoseCalculator",
         description="Calculate medication dose based on patient weight in kg. Use when asked about dosing.",
         func=dose_calculator_fn),
]
```

### MCP Agent — Selecting the Right Tool

```python
test_queries = [
    ("What medications is patient 5 currently taking?",    "PatientLookup"),        # → vector search
    ("Does clarithromycin interact with clopidogrel?",     "DrugInteractionCheck"), # → rule lookup
    ("Calculate metformin dose for a 94kg patient",        "DoseCalculator"),        # → math
    ("Which patient has renal problems?",                  "PatientLookup"),
    ("Is warfarin safe with aspirin?",                     "DrugInteractionCheck"),
]

for query, expected_tool in test_queries:
    tool_choice = llm.invoke([
        SystemMessage(content=f"Tool router. Respond with ONE tool name only.\nTools:\n{tool_descriptions}"),
        HumanMessage(content=query)
    ]).content.strip()

    selected = next(t for t in tools if t.name.lower() in tool_choice.lower())
    result = selected.func(query)
    correct = expected_tool.lower() in tool_choice.lower()
    print(f"Q: {query}")
    print(f"   Picked: {tool_choice} | Expected: {expected_tool} | {'✓' if correct else '✗'}")
```

### Full MCP Flow — Combining Two Tools

```python
# Complex query needs PatientLookup AND DrugInteractionCheck
complex_q = "Patient 5 is on clarithromycin and clopidogrel. Is this combination safe?"

patient_data = tools[0].func("patient 5 medications")       # Tool 1: RAG
interaction   = tools[1].func("clarithromycin clopidogrel") # Tool 2: rule-based

final_answer = llm.invoke([
    SystemMessage(content=f"Clinical assistant. Use these tool outputs:\n\nPatient Data:\n{patient_data}\n\nDrug Interaction:\n{interaction}"),
    HumanMessage(content=complex_q)
]).content

# Expected: "UNSAFE. Clarithromycin inhibits CYP3A4, reducing Clopidogrel to its active form.
#            This significantly increases cardiac event risk for Patient 5."
```

> "RAG answered 'what drugs?' (retrieval from PDFs). Drug interaction check answered 'is it safe?' (rule lookup — NOT retrieval). The agent used both through one protocol. That is MCP: one protocol, many tools, the agent picks the right one."

---

## 9. Conversational Agent Capstone — ClinicalAssistant

The instructor put everything together: PDF knowledge + conversation memory in one persistent agent.

```python
class ClinicalAssistant:
    '''Multi-turn agent: retrieves from PDFs + remembers conversation.'''

    def __init__(self, pdf_store):
        self.pdf_store = pdf_store    # static knowledge (PDFs)
        self.memory_store = None      # grows with each conversation turn
        self.turn_count = 0

    def _add_memory(self, text):
        if self.memory_store is None:
            self.memory_store = FAISS.from_texts([text], embeddings)
        else:
            self.memory_store.add_texts([text])

    def chat(self, message):
        self.turn_count += 1

        # 1. Retrieve from knowledge (PDFs)
        pdf_results = self.pdf_store.similarity_search(message, k=3)
        pdf_context = "\n".join([d.page_content for d in pdf_results])

        # 2. Retrieve from conversation history
        memory_context = "(No prior conversation)"
        if self.memory_store:
            mem_results = self.memory_store.similarity_search(message, k=3)
            if mem_results:
                memory_context = "\n".join([d.page_content for d in mem_results])

        # 3. Generate response with both contexts
        system = f'''You are a clinical decision support assistant.

Patient Data (from uploaded PDFs):
{pdf_context}

Conversation History:
{memory_context}

Rules: cite patient names, medication names, lab values. Reference prior conversation if relevant.'''

        response = llm.invoke([
            SystemMessage(content=system),
            HumanMessage(content=message)
        ]).content

        # 4. Store this turn into memory
        record = f"[Turn {self.turn_count}] User: {message} | Assistant: {response[:150]}"
        self._add_memory(record)

        return response
```

### 5-Turn Demo Conversation

```python
assistant = ClinicalAssistant(pdf_store)

turns = [
    "Tell me about patient 5 — the cardiac patient.",          # Turn 1
    "What medications is he currently on?",                    # Turn 2: "he" resolved via memory
    "Are there any dangerous drug interactions I should know?", # Turn 3
    "What about patient 3 — the one with kidney problems?",    # Turn 4: topic switch
    "Comparing patients 3 and 5, who is at higher immediate risk?", # Turn 5: cross-doc
]

for msg in turns:
    response = assistant.chat(msg)
    print(f"You: {msg}")
    print(f"AI:  {response[:300]}\n")
```

**Expected outputs:**
- Turn 1: Identifies David Okafor (Patient 5), cardiac diagnosis
- Turn 2: Lists Clopidogrel 75mg, Clarithromycin, Atorvastatin
- Turn 3: Flags CYP3A4 SEVERE interaction (Clarithromycin inhibits Clopidogrel activation)
- Turn 4: Switches to Robert Chen (Patient 3), CKD Stage 3, eGFR 42, Metformin caution
- Turn 5: Compares both — Patient 5 = higher **immediate** risk (life-threatening drug interaction)

**Memory test (Turn 6):**
```python
memory_test = assistant.chat("What did I ask you about first in this conversation?")
# Expected: "You first asked about Patient 5, the cardiac patient named David Okafor."

cross_test = assistant.chat("Which patient's drugs interact with each other?")
# Expected: "Patient 5 — the Clarithromycin + Clopidogrel combination is a severe CYP3A4 interaction."
```

---

## 10. Group Exercise — Build Your Domain RAG Agent

Participants were given 25 minutes to build a domain-specific RAG agent using the capstone template.

**Template structure:**
1. Replace `my_knowledge_base` with 10–15 domain-specific chunks (each 1–3 sentences on ONE topic)
2. Build FAISS store
3. Instantiate `MyDomainAgent` (RAG + memory)
4. Test with 3 questions
5. Present to class (3 minutes)

> "The more specific your chunks, the better the retrieval. Don't write paragraphs — write one clear fact per chunk."

---

## Key Takeaways

| Concept | Summary |
|---------|---------|
| **RAG Pipeline** | `RetrievalQA.from_chain_type()` with `chain_type="stuff"` — inject top-k chunks into prompt |
| **Hallucination test** | RAG cites the actual policy; bare LLM invents plausible but wrong answers |
| **PDF ingestion** | PyPDFLoader → RecursiveCharacterTextSplitter(800, 100) → FAISS.from_documents |
| **Chunking overlap** | overlap=100 prevents losing context at chunk boundaries |
| **Memory loop** | `store.add_texts([record])` after each response adds Q&A back to FAISS |
| **Multi-store RAG** | VDB1 (knowledge) + VDB2 (thumbs up) + VDB3 (thumbs down) = RLHF at inference |
| **Saturation point** | As VDB2/3 grow, agent approaches exact-match retrieval and stops repeating mistakes |
| **Validation gate** | LLM confidence >95% before adding to memory — customers are wrong ~20% of the time |
| **FAISS persistence** | `store.save_local("path")` + `FAISS.load_local("path", embeddings, allow_dangerous_deserialization=True)` |
| **MCP ≠ RAG** | MCP is a protocol (like HTTP) for AI-to-tool communication; RAG is one tool in MCP |
| **MCP 3 layers** | Host (app) → Client (connection manager) → Server (exposes tools/resources/prompts) |
| **Domain-specific servers** | Avoid one mega-server; use separate servers per function (support, sales, marketing) |
| **Tool descriptions** | LLM selects tools by description alone — bad description = wrong tool called every time |
| **Team split** | AI/ML = agent + descriptions; Software Engineering = MCP server + tool functions |
| **Capstone** | ClinicalAssistant: PDF knowledge store + conversation memory store → multi-turn grounded agent |

**The 5 Laws of Production Agent Memory (from notebook):**
1. **PERSISTENCE** — memory survives application restarts
2. **RETRIEVAL** — searchable by meaning, not just keyword
3. **DECAY** — old memories fade; you don't keep everything forever
4. **SEPARATION** — shared knowledge ≠ private conversation history
5. **GOVERNANCE** — who reads, writes, deletes? Audit trail required.

---

## Files in This Folder

| File | Purpose |
|------|---------|
| `Week12_19thApr_Memory & Knowledge Retrieval in Agents with MCP.md` | Session notes (this file) |
| `1 - 74409 - Live session on Memory and Knowledge Retrieval in Agents with Ramkrishna - TRANSCRIPT.txt` | WEBVTT transcript from session |
| `Week12_Day2_Notebook.ipynb` | Complete practical notebook — RAG pipeline, PDF ingestion, memory loop, MCP demo, capstone |
