# Week 10 – Agent Architectures & Collaboration
## Session: March 28, 2026 | 10:00 AM IST
**Presenter:** Abhinav Singh

---

## Session Overview

This session focused on:
- Clarifying previous homework: Insurance claim validation multi-agent system
- Multi-agent orchestration frameworks: LangChain, LangGraph, CrewAI
- Pharmaceutical use case: PubMed-based multi-agent pipeline (live coding)
- Drug interaction reasoning agent (Warfarin + Ibuprofen live demo)
- LLM selection strategy: SLM vs LLM, Instant vs Thinking models
- CrewAI Studio no-code visual builder (live demo)
- Industry applications and real-world agentic AI challenges
- Breakout room: Students built their own agent system designs

---

## 1. LLM & Platform Selection – Key Concepts

### 1.1 Groq vs LLM — Critical Distinction

This was a common source of confusion from the previous week.

**Question asked by student:** "What exactly is Groq? Is it the same as the LLM?"

**Ramakrishna's answer:**
- **Groq** is a **hardware-based inference platform** — it provides extremely fast inference using LPU (Language Processing Unit) chips
- Groq is **NOT a model** — it is a hosting/serving platform
- The **LLM** being used is **Llama 3.1 8B Instant**, an open-source model created by Meta
- Groq hosts this model and provides an API endpoint that is API-compatible with OpenAI
- So when you call `ChatGroq(model="llama-3.1-8b-instant")`, you are calling Meta's Llama model served via Groq's fast inference infrastructure

> "Think of Groq as AWS — it's the cloud provider. The LLM is like an application running on that cloud. Llama 3.1 is the application. Groq is the cloud."

### 1.2 When to Use Small vs Large Language Models

| Criteria | Small Language Model (SLM) | Large Language Model (LLM) |
|----------|---------------------------|---------------------------|
| Task complexity | Simple tasks (crawling, formatting, routing) | Complex reasoning tasks |
| Cost | Low / Free (especially with Groq) | Higher API cost |
| Speed | Faster (Instant models) | Slower |
| Reasoning | Limited | Strong |
| Examples | Llama 8B Instant, Gemini Flash | GPT-4, Claude Opus, Gemini Pro |

**Rule of thumb from Ramakrishna:**
1. Always start with SLMs
2. Test against a **golden dataset** (labeled test cases with known correct answers)
3. Only upgrade to LLMs if accuracy is insufficient
4. Don't pay for LLM compute unless you have proven you need it

> "In production, you could use Llama 8B for query decomposition and routing, but switch to Claude Opus for the clinical reasoning step that requires deep medical logic. You don't have to use the same model everywhere."

### 1.3 Instant Models vs Thinking Models

| Type | Example Models | Characteristic | Best For |
|------|----------------|----------------|----------|
| Instant | Llama 8B Instant, Gemini Flash | Fast, low latency, shallow reasoning | Data fetching, formatting, routing steps |
| Thinking | GPT-4, Claude Opus, Gemini Pro | Slow, deep reasoning, expensive | Medical diagnosis, legal analysis, strategic planning |

**Key insight:** The same agent pipeline can mix models — use instant models for lightweight steps and thinking models for the step that requires real intelligence.

---

## 2. Recap: Insurance Claim Validation Homework

### 2.1 Problem Statement

Students were given a homework task to build a multi-agent insurance claim validation system.

**Input:**
- Claim ID
- User's natural language query
- Policy ID

**Output:**
- Decision: `Approve` / `Reject` / `Partial`
- Reason (what policy clauses were violated)
- Mismatch details (if any)

### 2.2 Sample Claims Data

```python
claims_data = {
    "C101": {
        "incident": "Car accident",
        "reported_days": 2,        # reported within 7-day window — OK
        "amount_claimed": 3000,    # within policy limit — OK
        "history": "No previous claims"
    },
    "C102": {
        "incident": "Car accident",
        "reported_days": 12,       # VIOLATION: reported after 7-day window
        "amount_claimed": 4000,    # may exceed limit depending on policy
        "history": "2 previous claims"   # risk factor
    },
    "C103": {
        "incident": "Theft",
        "reported_days": 1,        # within window — OK
        "amount_claimed": 6000,    # may exceed theft coverage limit
        "history": "No previous claims"
    }
}
```

### 2.3 Multi-Agent Architecture for Insurance

```
User (Claim ID + Query + Policy ID)
         │
         ▼
[Agent 1: Database Fetcher]
    Fetches claim record from claims_data
         │
         ▼
[Agent 2: PDF Reader]
    Reads the policy PDF, extracts relevant clauses
         │
         ▼
[Agent 3: Comparator]
    Compares claim attributes vs policy rules
         │
         ▼
[Agent 4: Decision Maker]
    Returns: Decision + Reason + Mismatch
```

### 2.4 Example Decision Output

For **C102** (reported 12 days after incident):
```
Decision: Reject
Reason:
  - Claim C102 was reported 12 days after the incident
  - Policy requires reporting within 7 days of the incident
  - Amount claimed ($4,000) also exceeds the single-incident limit
Recommendation: Reject. Partial payout may be considered at discretion.
```

### 2.5 Key Learning from Homework Discussion

- **Policy documents are long** — 5 to 100+ pages in real insurance companies
- The agent must extract only the *relevant clauses* for the given claim type
- **Data simulation:** When real company data isn't available, use ChatGPT to generate synthetic sample claims and policies — this is a legitimate data science practice
- Ramakrishna mentioned he regularly uses LLMs to generate 50-100 synthetic test records before real data becomes available

---

## 3. Agentic Frameworks Overview

| Framework | Description | Best For |
|-----------|-------------|----------|
| **LangChain** | Core orchestration: tools, agents, chains, retrievers | Building single/multi-agent pipelines, RAG, tool use |
| **LangGraph** | Graph-based agent flow, stateful multi-step agents | Complex stateful workflows, conditional branching |
| **AutoGen** | Microsoft's multi-agent conversation framework | Conversational agents that talk to each other |
| **CrewAI** | High-level multi-agent framework with roles, goals, tasks | Teams of specialized agents with minimal code |

### 3.1 CrewAI – Key Highlights

- You define agents with **Role + Goal + Backstory + Task + Expected Output**
- CrewAI handles the orchestration logic automatically
- Supports both **sequential** and **hierarchical** orchestration modes
- Has a **Crew Manager** component that decides which agent to call when in hierarchical mode
- Can be built via Python code OR via the no-code **CrewAI Studio** visual interface
- Supports scheduling (daily/weekly cron-style runs)
- Can be deployed as an API endpoint

---

## 4. Pharmaceutical Multi-Agent Use Case (PubMed Pipeline)

### 4.1 Motivation — Why Agents Over Plain LLMs?

**Student question:** "Why can't we just ask GPT-4 about drug interactions directly?"

**Ramakrishna's answer (detailed):**
1. **Knowledge cutoff**: LLMs have training data cutoffs (e.g., April 2023). New drug studies from the past year won't be in the model's memory.
2. **Hallucination risk**: LLMs can confidently state wrong dosages or interactions that don't exist.
3. **No real-time access**: LLMs can't query live databases — they can't check if a drug was recalled last month.
4. **Internal data**: LLMs can't access your hospital's EHR (Electronic Health Records) system.
5. **Verifiability**: An agent that cites PubMed papers gives you traceable evidence; a plain LLM just says "trust me."

> "In healthcare, you cannot just say 'GPT-4 told me this.' You need a citation. You need to be able to say 'this is from PubMed paper ID 12345, published by Johns Hopkins in 2024.'"

### 4.2 Why Not RAG (Retrieval-Augmented Generation)?

**Student question:** "Couldn't we build a RAG system on PubMed?"

**Ramakrishna's answer:**
- PubMed has **40 million papers** — building a vector DB from all of them is computationally and financially impractical
- Papers are constantly being added — any static vector DB would go stale within weeks
- PubMed already provides a free, well-maintained search API (NCBI E-Utilities) — no need to reinvent it
- The agent approach calls the live API at query time, always getting fresh results
- RAG is appropriate when you have a bounded, static document set (e.g., your company's internal policy docs)

### 4.3 PubMed API — Technical Details

**PubMed** = National Library of Medicine (NCBI) database with ~40 million biomedical citations

**Step 1: Search for paper IDs**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
```
Parameters:
- `db=pubmed` — the database
- `term=warfarin ibuprofen interaction` — your search query
- `retmax=3` — how many results to return (keep low to avoid flooding context)
- `retmode=json` — response format

Returns: A list of PubMed paper IDs (integers)

**Step 2: Fetch paper abstracts**
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
```
Parameters:
- `db=pubmed`
- `id=12345678,87654321,11111111` — comma-separated IDs from Step 1
- `retmode=text`
- `rettype=abstract` — return just the abstract (not full paper)

Returns: Plain text abstracts (trimmed to ~2000 chars per paper to stay within LLM context limits)

**Code implementation from `practice.ipynb`:**
```python
import requests

def pubmed_search(query, max_results=3):
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"

    # Step 1: Get paper IDs
    res = requests.get(
        base + "esearch.fcgi",
        params={"db": "pubmed", "term": query, "retmax": max_results, "retmode": "json"}
    ).json()
    ids = res.get("esearchresult", {}).get("idlist", [])

    # Step 2: Fetch abstracts
    abstracts = requests.get(
        base + "efetch.fcgi",
        params={"db": "pubmed", "id": ",".join(ids), "retmode": "text", "rettype": "abstract"}
    ).text

    return abstracts[:2000]  # Truncate to protect LLM context window
```

### 4.4 5-Agent Pipeline Architecture

```
User Query: "Patient taking Warfarin and Ibuprofen. Risks? Alternatives?"
         │
         ▼
[Agent 1: Query Decomposition]
    Breaks query into 3 PubMed-optimized search terms:
    - "warfarin ibuprofen drug interaction"
    - "NSAIDs anticoagulant bleeding risk"
    - "warfarin alternatives pain management"
         │
         ▼
[Agent 2: PubMed Retrieval]
    Calls pubmed_search() for each of the 3 queries
    Returns: 3 sets of abstracts (up to 9 papers total)
         │
         ▼
[Agent 3: Evidence Summarization]
    LLM reads all abstracts
    Extracts: key risks, mechanisms, findings, drug names
    Output: Structured evidence summary
         │
         ▼
[Agent 4: Clinical Reasoning]
    Takes evidence summary + original query
    Provides:
    - Interaction risk level (e.g., HIGH)
    - Mechanism of interaction
    - Safer alternatives
    - Clinical recommendation
         │
         ▼
[Agent 5: Validator]
    Checks for:
    - Hallucinations (claims not supported by abstracts)
    - Unsafe advice (e.g., suggested dosage without citing evidence)
    - Missing warnings (e.g., forgot to mention monitoring INR)
    - Flags any issues back to Agent 4 for revision
         │
         ▼
Final Clinical Response (with citations)
```

### 4.5 Sample Output — Warfarin + Ibuprofen

```
DRUG INTERACTION ANALYSIS

1. Interaction Risk: HIGH

2. Mechanism:
   - Ibuprofen (NSAID) inhibits platelet aggregation via COX-1 inhibition
   - Ibuprofen disrupts gastric mucosal protection
   - Warfarin inhibits clotting factor synthesis (Vitamin K antagonist)
   - Combined effect: significantly elevated gastrointestinal bleeding risk
   - Some NSAIDs also inhibit warfarin metabolism, increasing INR

3. Safer Alternatives:
   - Acetaminophen (Paracetamol) — first-line; does not affect platelet aggregation
   - Naproxen — use with caution; still an NSAID but lower GI risk than ibuprofen
   - Celecoxib (COX-2 inhibitor) — lower GI bleeding risk than traditional NSAIDs

4. Clinical Recommendation:
   - Avoid ibuprofen in patients on Warfarin
   - Monitor INR regularly if any NSAID must be used
   - Monitor CBC for signs of GI bleeding
   - Prefer acetaminophen as first-line analgesic

5. Source: PubMed papers [IDs shown] — 3 studies retrieved
```

### 4.6 Multi-Database Strategy (Discussed During Q&A)

**Student question:** "What if PubMed doesn't have the information?"

**Answer:** Build separate tools for each medical database and let the agent decide which to query:

| Database | What It Contains | API Available? |
|----------|-----------------|----------------|
| PubMed (NCBI) | 40M biomedical papers, abstracts | Yes — free |
| OpenFDA | FDA drug approvals, recalls, adverse events | Yes — free |
| RxNorm | Drug names, synonyms, interaction IDs | Yes — free |
| EMBASE | European medical literature | Paid |
| Google Scholar | Broad academic literature | Limited (unofficial) |
| PMC (PubMed Central) | Full-text papers (not just abstracts) | Yes — free |

> "Each of these becomes a separate tool in your agent. The agent decides which tool to call. If PubMed returns nothing useful, it tries RxNorm. This is why tool selection is powerful."

---

## 5. LangChain Code Patterns (from practice.ipynb)

### 5.1 LLM Setup

```python
import os
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0  # 0 = deterministic output
)
```

### 5.2 Simple LLM Call vs Agent

```python
# Simple LLM call — NOT an agent
response = llm.invoke("What is the capital of France?")
print(response.content)

# Agent — has tools, reasoning loop
from langchain.agents import AgentExecutor, create_react_agent

agent = create_react_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, max_iterations=5, verbose=True)
result = agent_executor.invoke({"input": "Find risks of Warfarin + Ibuprofen"})
```

### 5.3 max_iterations — Why It Matters

```python
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=5,   # agent can take up to 5 actions before giving final answer
    verbose=True
)
```

**Ramakrishna's explanation of what happens internally:**

With `max_iterations=5`, the agent loop looks like:
```
Iteration 1: Thought → calls pubmed_search("warfarin ibuprofen")
Iteration 2: Thought → calls pubmed_search("NSAIDs anticoagulant")
Iteration 3: Thought → processes results, calls summarize tool
Iteration 4: Thought → calls reasoning tool
Iteration 5: Final Answer → returns clinical response
```

**Debugging tip:** If your agent isn't checking all data sources or seems to give up too early:
1. First suspect: `max_iterations` is too low
2. Increase by 1-2 and test again
3. Don't always blame the prompt — the agent may physically not have enough iterations

### 5.4 Multi-Agent as Sequential LLM Calls

**Kiran (student) asked:** "Is this really a multi-agent system or just sequential LLM calls?"

**Ramakrishna's answer:**
- Technically, in many implementations, "agents" are just sequential LLM invocations
- What makes it "agentic" is:
  1. **Autonomy** — the agent decides *which* tool to call based on context
  2. **Communication** — each agent's output is passed as context to the next
  3. **Goal-directedness** — the system works toward a high-level goal, not just executing pre-defined steps
- You don't always need `AgentExecutor` — you can chain `llm.invoke()` calls and still call it a multi-agent system if there is specialization and communication

---

## 6. CrewAI Studio — No-Code Visual Builder (Live Demo)

### 6.1 What is CrewAI Studio?

CrewAI Studio is a **visual, no-code interface** for building multi-agent systems.

Instead of writing Python code, you:
1. Describe your use case in plain English
2. CrewAI Studio auto-generates agents with roles, descriptions, expected outputs
3. You review/edit the generated configuration
4. Run the crew directly from the UI

**URL:** app.crewai.com (requires account)

### 6.2 Demo: Pharmaceutical Research Agent (Live)

Abhinav Singhyped the following into CrewAI Studio's prompt:

> *"Build a pharmaceutical research agent that takes a drug interaction query, searches PubMed, summarizes evidence, provides clinical recommendations, and sends a summary email to the researcher."*

**Auto-generated output from CrewAI Studio:**

**Agents created:**
1. `PubMed Research Specialist`
   - Role: Biomedical literature researcher
   - Goal: Search PubMed for relevant studies on given drug combinations
   - Tools: Web search, PubMed API tool

2. `Evidence Synthesis Expert`
   - Role: Summarize and structure evidence from multiple abstracts
   - Goal: Produce clear, structured summary of findings

3. `Clinical Pharmacology Advisor`
   - Role: Provide clinical recommendations based on evidence
   - Goal: Advise on interaction risks, mechanisms, safer alternatives

4. `Email Communication Agent`
   - Role: Send formatted research summary via Gmail
   - Tools: Gmail integration

**Tasks created:**
1. Search PubMed for drug interaction papers
2. Summarize findings into structured evidence report
3. Generate clinical recommendation
4. Send email with attached report

### 6.3 CrewAI Agent Configuration Options

When you click on any agent in CrewAI Studio, you can configure:

| Setting | Options | Notes |
|---------|---------|-------|
| LLM Model | GPT-4, Claude Opus, Llama 3, Gemini, etc. | Per-agent model selection |
| Temperature | 0.0 – 1.0 | 0 = deterministic, 0.7 = balanced, 1.0 = creative |
| Max Iterations | 1–20 | How many tool calls the agent can make |
| Max Requests/Min | Rate limit | Prevents hitting API quotas |
| Response Format | Text / JSON / Structured | Force structured output if needed |
| Tools | From tool library | Gmail, Calendar, web scraping, custom tools |

### 6.4 Temperature — Explained in Detail

**Student question:** "What temperature should we use? What does it actually do?"

**Ramakrishna's answer:**
- Temperature controls **randomness in token selection** at each generation step
- **0** = always picks the highest-probability next token → same output every time → good for deterministic tasks (data extraction, code generation)
- **1.0** = full random sampling → creative but unpredictable → good for creative writing, brainstorming
- **0.6–0.7** = recommended for most business use cases — some variation but still coherent and accurate
- In production agents: use **0 for data-fetching/parsing steps** and **0.6-0.7 for generation/reasoning steps**

### 6.5 Tools in CrewAI

**Out-of-the-box tools available:**
- Gmail (read/send emails)
- Google Calendar (create/read events)
- Web scraping (fetch URL content)
- File read/write
- Custom Python function tools

**Adding a custom tool:**
```python
# customtool.py
from crewai_tools import BaseTool

class PubMedTool(BaseTool):
    name: str = "PubMed Search"
    description: str = "Searches PubMed for biomedical papers given a query"

    def _run(self, query: str) -> str:
        # Your pubmed_search() function here
        return pubmed_search(query)
```

### 6.6 Human-in-the-Loop (HITL) in CrewAI Studio

**Toggle options for the Email agent:**
- **"Create email draft"** — agent prepares draft, human reviews before sending
- **"Send email directly"** — agent sends without human confirmation

> "In production, for anything that sends messages, makes purchases, or modifies external systems — always use 'draft first' mode. You can always remove the human checkpoint later once you've built trust in the system."

### 6.7 Scheduling / Triggers

CrewAI Studio supports:
- **Manual runs** — click "Run" in UI
- **Scheduled runs** — cron-style (daily at 8 AM, weekly on Monday)
- **Trigger-based** — webhook URL that external systems can call

Use case demonstrated:
- Pharma research digest: runs every Monday 6 AM, fetches latest PubMed papers on 10 tracked drug pairs, emails summary to clinical team

### 6.8 Deployment as API

Once your crew is ready:
1. Click **"Publish"** in CrewAI Studio
2. Get an **API URL** + **Bearer Token**
3. Call it from any frontend:

```python
import requests

response = requests.post(
    "https://api.crewai.com/v1/crews/your-crew-id/run",
    headers={"Authorization": "Bearer YOUR_TOKEN"},
    json={"inputs": {"query": "Warfarin + Ibuprofen risks"}}
)
print(response.json())
```

> "Now your crew is a microservice. Your React app, your mobile app, your Slack bot — anything can call it. You've just deployed a multi-agent AI system without writing a single line of deployment code."

---

## 7. Industry Context — Real-World Applications

### 7.1 Pharma Industry Gap

Ramakrishna mentioned receiving a real inquiry from a pharmaceutical company startup:
- They were building a **research intelligence product** for Novartis/Pfizer researchers
- Researchers currently spend 30-40 minutes manually searching PubMed for each drug query
- The startup wanted to automate this with a multi-agent pipeline exactly like the one demonstrated
- This is a multi-million dollar market opportunity

> "If you can build this system correctly — with proper citations, hallucination checking, regulatory-safe language — you have a real product. This is not a toy demo."

### 7.2 95% of AI Work is at the Application Layer

**Ramakrishna's career advice:**
- 95% of companies hiring AI engineers work at the **application layer**
- You are calling LLM APIs, building tools, tuning prompts, integrating with databases
- Only Anthropic, OpenAI, Google, Meta are training foundation models
- Your value: **knowing which architecture to use, which model to pick, how to prompt effectively, how to debug agent behavior**
- Claude/Copilot can give you 60–70% of code; the remaining 30% is your domain knowledge + engineering judgment

---

## 8. Breakout Room Sessions

### 8.1 Exercise Given to Students

> "Open CrewAI Studio. Build a multi-agent system for YOUR domain using what you learned today. You have 15 minutes. Then present back."

### 8.2 George's Presentation — Guitar Learning Platform

**Domain:** Music education

**System built:**
- **Agent 1: Skill Assessor** — Asks student questions, identifies current level (beginner/intermediate)
- **Agent 2: Curriculum Designer** — Creates personalized 8-week guitar learning roadmap
- **Agent 3: Progress Tracker** — Monitors completed lessons, sends weekly progress report
- **Agent 4: Email Agent** — Sends weekly email with next lesson plan + YouTube links

**Ramakrishna's feedback:**
- Good application of sequential flow
- Suggested adding a **feedback loop agent** — if student reports difficulty, curriculum adjusts
- Noted that this is a real product opportunity in the edtech space

### 8.3 Kiran's Presentation — Smart Flight Booking Agent

**Domain:** Travel / Flight booking

**System built:**
- **Agent 1: Query Parser** — Understands natural language: "I need to fly from Mumbai to London next Friday"
- **Agent 2: Flight Search Agent** — Searches Amadeus/Skyscanner API for available flights
- **Agent 3: Price Optimization Agent** — Checks multiple dates ±3 days for better prices
- **Agent 4: Booking Agent** — Confirms booking (with HITL checkpoint before payment)

**Kiran's question:** "Is this a goal-oriented agent? Because it takes multiple steps?"
**Answer:** Yes — this is a perfect goal-oriented agent. The goal is "book the best flight," and the agent takes multiple steps to achieve it. Single-step would be just "search for flights."

**Ramakrishna's feedback:**
- Good architecture
- Suggested adding a **price alert agent** — if price drops after search, notify the user
- Noted that the ±3 days price optimization step is exactly how Google Flights works internally

### 8.4 Publishing/Deployment Discussion (Post-Breakout)

**Student question:** "After we build this in CrewAI Studio, how do we give it to clients?"

**Ramakrishna's answer (step-by-step):**
1. **CrewAI Studio → Publish → Get API URL + Token**
2. **Build a simple frontend** (HTML + JavaScript, or React, or Streamlit)
3. Frontend calls your CrewAI API with user's input
4. Display the results in the UI
5. You can also embed the agent in a Slack bot, WhatsApp, or website chat widget

> "The agent is now your backend. You're a full-stack AI developer. The hard part was building the agent — deploying it is just a POST request."

---

## 9. Key Takeaways

| Concept | Summary |
|---------|---------|
| Groq | Inference platform, NOT a model. Hosts Llama and other models with fast LPU hardware. |
| SLM vs LLM | Start with SLMs (Llama 8B, Gemini Flash). Upgrade to LLMs only if accuracy is insufficient after testing. |
| Instant vs Thinking | Instant = fast, low reasoning. Thinking = slow, deep reasoning. Mix both in a pipeline. |
| Why agents > LLMs | Knowledge cutoff, hallucination risk, no real-time data access, no internal DB access, no verifiability |
| Why agents > RAG for PubMed | 40M+ papers = impractical for vector DB. Live API is better. |
| PubMed API | esearch.fcgi → get IDs; efetch.fcgi → get abstracts. Free, no auth needed. |
| 5-Agent Pharma Pipeline | Decompose → Retrieve → Summarize → Reason → Validate |
| Multiple medical DBs | PubMed + OpenFDA + RxNorm + PMC = different tools for same agent |
| Temperature | 0 = deterministic, 0.6-0.7 = recommended for business use, 1.0 = creative |
| max_iterations | Controls how many tool calls before giving final answer. Too low = agent gives up early. |
| CrewAI Studio | No-code visual builder. Auto-generates roles, tasks, expected outputs from plain English. |
| HITL | Use "draft" mode for any agent that sends messages or modifies external systems. |
| Deployment | Publish → API URL + Bearer Token → call from any frontend. Agent becomes a microservice. |
| Career advice | 95% of AI work = application layer. Domain knowledge + prompt engineering + architecture choice = your value. |

---

## 10. Files in this Folder

| File | Description |
|------|-------------|
| `16675 - Ramakrishna - 28-03-2026.zip` | Shared materials from session |
| `28thMar_extracted/practice.ipynb` | PubMed multi-agent notebook with Gradio UI |
| `28thMar_extracted/crewai.txt` | 7 domain problem statements for CrewAI practice |
| `28thMar_extracted/LangChain_HomeWork.txt` | Insurance claim validation homework |
| `Week 10_ Agent Architectures & Collaboration.pdf` | Session slides |
| `0 - 60613 - ... TRANSCRIPT.txt` | Full session transcript (WEBVTT format) |

---

## 11. Appendix: 7 CrewAI Domain Practice Problems (from crewai.txt)

| # | Domain | Agent Task |
|---|--------|------------|
| 1 | Finance | Investment Risk Analyst — research company + risk summary + email to advisor |
| 2 | E-commerce | Smart Purchase Advisor — compare products + track price trend + alert on drop |
| 3 | DevOps | Incident Intelligence — monitor alerts + summarize root cause + notify team |
| 4 | Healthcare | Hospital Operations — analyze patient bottlenecks + recommend workflow changes |
| 5 | News/Research | Topic Intelligence — track topic daily + summarize + flag emerging risks |
| 6 | Supply Chain | Delay Prediction — gather signals + predict disruption + alert procurement |
| 7 | Education | Learning Recommendation — build roadmap + identify risk + email weekly plan |

**Universal Agent Blueprint (applies to all 7):**
1. Understand user query / intent
2. Generate optimized search queries
3. Fetch external or internal data (API / DB / PDF)
4. Summarize findings into structured insights
5. Identify risks, decisions, or action items
6. Trigger action: email / Slack / Teams / calendar entry
