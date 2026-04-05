# Week 10 – Agent Architectures & Collaboration
## Session: March 28, 2026 | 10:00 AM IST
**Presenter:** Ramakrishna T

---

## Session Overview

This session focused on:
- Clarifying previous homework: Insurance claim validation multi-agent system
- Multi-agent orchestration frameworks: LangChain, LangGraph, CrewAI
- Pharmaceutical use case: PubMed-based multi-agent pipeline
- Drug interaction reasoning agent (Warfarin + Ibuprofen demo)
- LLM selection strategy: SLM vs LLM
- Industry applications and real-world agentic AI challenges

---

## 1. LLM & Platform Selection – Key Concepts

### Groq vs LLM — Important Distinction
- **Groq** is a **platform/inference service**, NOT a model
- Groq hosts both open-source models (e.g., Llama) and proprietary models
- The **LLM** used in previous weeks was **Llama 3.1 8B Instant** (hosted on Groq)

### When to Use Small vs Large Language Models

| Criteria | Small Language Model (SLM) | Large Language Model (LLM) |
|----------|---------------------------|---------------------------|
| Task complexity | Simple tasks (crawling, formatting) | Complex reasoning tasks |
| Cost | Low / Free | Higher API cost |
| Speed | Faster (Instant models) | Slower |
| Reasoning | Limited | Strong |
| Examples | Llama 8B Instant, Gemini Flash | GPT-4, Claude Opus, Gemini Pro |

> **Rule of thumb:** Start with SLMs. Only upgrade to LLMs when accuracy is insufficient. Always measure against a **golden dataset** before deciding.

### Instant Models vs Thinking Models
- **Instant models** (e.g., Llama 8B Instant, Gemini Instant): Fast, low reasoning, for straightforward tasks
- **Thinking models** (e.g., GPT-4, Claude Opus): Slower, deep reasoning, for complex decisions
- Always experiment — latency vs accuracy is the trade-off

---

## 2. Recap: Insurance Claim Validation Agent (Homework)

### Problem Statement (from LangChain_HomeWork.txt)
Build an AI system where:
- **User provides:** Claim ID, Query, Policy ID
- **System should:**
  1. Fetch claim details from the database
  2. Read the policy document (PDF)
  3. Compare policy rules vs. claim attributes
  4. Return: Decision, Reason, Mismatch (if any)

### Sample Claims Data
```python
claims_data = {
    "C101": {"incident": "Car accident", "reported_days": 2, "amount_claimed": 3000, "history": "No previous claims"},
    "C102": {"incident": "Car accident", "reported_days": 12, "amount_claimed": 4000, "history": "2 previous claims"},
    "C103": {"incident": "Theft", "reported_days": 1, "amount_claimed": 6000, "history": "No previous claims"}
}
```

### Tasks the Agent Must Perform
1. Is the claim valid?
2. Are there any violations? (e.g., claim reported after 7-day window)
3. Is there an amount mismatch?
4. Final decision: **Approve / Reject / Partial**

### Example Decision
```
Decision: Reject
Reason:
- Reported after allowed period (7 days)
- Claim exceeds policy limit
Recommendation: Partial payout OR reject
```

### Key Learning
- Policy documents can be 5–100+ pages
- The agent must extract relevant clauses per query
- Multi-agent flow: one reads DB, one reads PDF, one compares, one decides
- **Data simulation**: When real data isn't available, use LLMs to generate synthetic samples

---

## 3. Agentic Frameworks Covered

### Frameworks Overview
| Framework | Description |
|-----------|-------------|
| **LangChain** | Core orchestration, tools, agents, chains |
| **LangGraph** | Graph-based agent flow, stateful multi-step agents |
| **AutoGen** | Microsoft's multi-agent conversation framework |
| **CrewAI** | High-level multi-agent framework with roles, goals, tasks |

### CrewAI – Key Highlights
- You don't need to write a lot of Python — define agents with **roles + goals + prompts**
- Supports both **sequential** and **hierarchical** orchestration
- Has a **Crew Manager** component that decides which agent to call
- Suitable for teams of specialized agents working together

---

## 4. Pharmaceutical Multi-Agent Use Case (PubMed Pipeline)

### Use Case: Drug Interaction Reasoning
> **Query:** "Patient taking Warfarin and Ibuprofen. What are the risks and safer alternatives?"

### PubMed API – How It Works
- **PubMed** = National Center for Biotechnology Information (NCBI) database
- ~40 million citations from medical research papers
- Supports free API (NCBI E-Utilities)

#### API Endpoint Used
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi
```
**Parameters:**
- `db=pubmed`
- `term=<your query>`
- `retmax=3` (number of results)
- `retmode=json`

Then fetch abstracts via:
```
https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi
```

### 4-Agent Pipeline Architecture

```
User Query
    │
    ▼
[Agent 1: Query Decomposition]
    │  Breaks query into 3 PubMed-optimized search terms
    ▼
[Agent 2: PubMed Retrieval]
    │  Fetches abstracts for each search query
    ▼
[Agent 3: Evidence Summarization]
    │  Extracts key risks, findings, insights from abstracts
    ▼
[Agent 4: Clinical Reasoning]
    │  Provides: Interaction Risk, Mechanism, Safer Alternatives, Recommendation
    ▼
[Agent 5: Validator]
    │  Checks for hallucinations, unsafe advice, missing warnings
    ▼
Final Clinical Response
```

### Sample Output (Warfarin + Ibuprofen)
**1. Interaction Risk:** Increased gastrointestinal bleeding risk
**2. Mechanism:** Ibuprofen inhibits platelet aggregation + disrupts GI mucosa; Warfarin inhibits clotting factors → additive bleeding risk
**3. Safer Alternatives:** Acetaminophen, Naproxen (with caution), Celecoxib
**4. Recommendation:** Avoid ibuprofen; monitor INR and CBC regularly; use acetaminophen as first-line alternative

---

## 5. Industry Context & Real-World Applications

### CrewAI Demo: Research Summary + Email/Calendar
- Agents can: search web → summarize → send email → add calendar entry
- All orchestrated without manual Python for each step

### 7 Domain Practice Problems (from crewai.txt)

| # | Domain | Agent Task |
|---|--------|------------|
| 1 | Finance | Investment Risk Analyst — research + risk summary + email |
| 2 | E-commerce | Smart Purchase Advisor — compare + price trend + alert |
| 3 | DevOps | Incident Intelligence — monitor + summarize + notify team |
| 4 | Healthcare | Hospital Operations — analyze bottlenecks + recommend |
| 5 | News/Research | Topic Intelligence — track + summarize + flag risks |
| 6 | Supply Chain | Delay Prediction — signals + predict + alert |
| 7 | Education | Learning Recommendation — roadmap + risk + email plan |

### Agent Blueprint (applies to all domains)
1. Understand user query
2. Generate search queries
3. Fetch external/internal data
4. Summarize insights
5. Identify risks / decisions
6. Trigger action (email / Slack / Teams)

---

## 6. Tools & Code Reference

### Libraries Used (from practice.ipynb)
```bash
pip install langchain langchain-groq langchain-community tavily-python python-dotenv
```

### LLM Setup
```python
from langchain_groq import ChatGroq
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    api_key=os.getenv("GROQ_API_KEY"),
    temperature=0
)
```

### Gradio UI — Clinical AI Assistant
- Built a live Gradio interface for the PubMed clinical agent
- Supports: text query + PDF upload + voice input
- Shows: System thinking logs + Final clinical response

---

## 7. Key Takeaways

- **Groq ≠ LLM** — it's an inference platform; models like Llama are the actual LLMs
- **SLM vs LLM** decision should be based on task complexity and cost — always benchmark
- **Instant models** are fast but limited in reasoning; use them for simple orchestration steps
- The **Insurance claim agent** is a perfect example of multi-agent orchestration with DB + PDF + comparison logic
- **PubMed API** is free and powerful for pharmaceutical/medical research agents
- **CrewAI** makes it easy to build multi-agent systems with minimal code using role-based prompts
- **Data simulation** using LLMs is a valid data science technique when real data is unavailable

---

## Files in this Folder

| File | Description |
|------|-------------|
| `16675 - Ramakrishna - 28-03-2026.zip` | Shared materials from session |
| `28thMar_extracted/practice.ipynb` | PubMed multi-agent notebook with Gradio UI |
| `28thMar_extracted/crewai.txt` | 7 domain problem statements for CrewAI practice |
| `28thMar_extracted/LangChain_HomeWork.txt` | Insurance claim validation homework |
| `Week 10_ Agent Architectures & Collaboration.pdf` | Session slides |
| `0 - 60613 - ... TRANSCRIPT.txt` | Full session transcript (WEBVTT format) |
