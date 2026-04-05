# IITM AI/ML Agentic Systems — Master Project Instructions

## Identity
- **Student:** Abhisss | IITM (Indian Institute of Technology Madras)
- **Course:** AI/ML Agentic Systems (Jan–Apr 2026) | Presenter: Ramakrishna T
- **Root:** `C:\Abhisss\IITM\AI_ML_Agentic\`
- **Branch:** Always `main`

---

## Folder Structure (exact canonical pattern)
```
WeekN_TopicName/
└── DDthMon/
    ├── WeekN_DDthMon_TopicName.md        ← PRIMARY NOTES (you write this)
    ├── DDthMon_extracted/                ← unzipped session materials
    │   ├── *.ipynb                       ← code notebooks
    │   ├── *.txt                         ← practice problems / homework
    │   └── Codes and data sets/          ← additional notebooks
    ├── 16675 - Ramakrishna - *.zip       ← raw zip from session
    └── 0 - NNNNN - ... TRANSCRIPT.txt   ← WEBVTT transcript (4000–7000 lines)
```

---

## Week → Topic Map
| Week | Dates | Folder Topic |
|------|-------|-------------|
| Week1 | Jan 17–18 | Getting Started with Python & ChatGPT |
| Week2 | Jan 24–25 | Data Types, Variables & Control Flow |
| Week3 | Jan 31–Feb 1 | Functions & Working with Libraries |
| Week4 | Feb 7–8 | Fundamentals of AI & ML |
| Week5 | Feb 14–15 | NLP & Text Processing |
| Week6 | Feb 21–22 | LLMs & Prompt Engineering |
| Week7 | Feb 28–Mar 1 | RAG & Vector Databases |
| Week8 | Mar 13 | Masterclass |
| Week9 | Mar 21–22 | LangChain & Agents |
| Week10 | Mar 28–29 | Agent Architectures & Collaboration |
| Week11 | Apr 4–5 | (upcoming) |

---

## ══════════════════════════════════════
## NOTE QUALITY CONTRACT (NON-NEGOTIABLE)
## ══════════════════════════════════════

Every session notes file MUST satisfy ALL of the following. A note that skips any item is **incomplete**.

### Mandatory Sections (in order)
1. **Session Overview** — bullet list of ALL topics covered (must match the actual transcript, not just the main demo)
2. **H2 section per major concept** — one section per distinct topic, numbered (1, 2, 3...)
3. **Q&A Exchanges** — student question → full instructor answer, verbatim, inside every relevant section
4. **Code Blocks** — every snippet shown or discussed, with language tag and inline comments explaining purpose
5. **Live Demo Documentation** — step-by-step what instructor did, what they typed, what was generated/returned
6. **Real-world analogies & examples** — every analogy the instructor used (these are the most memorable parts)
7. **Industry context** — any mention of companies, job roles, real products, startup examples
8. **Breakout room outcomes** — every student presentation + instructor feedback
9. **Key Takeaways** — summary table at end (Concept | Summary columns)
10. **Files in this Folder** — reference table of all files

### Completeness Standards
- A 2–3 hour session produces **350–700 lines** of notes. Shorter = something was missed.
- **Never summarize a Q&A** — capture the actual exchange. The confusion students had IS the learning.
- **Never skip "small" topics** — even a 2-minute tangent on a tool or concept gets its own subsection.
- **Never omit code** with "it's in the notebook" — the notes must be self-contained for exam review.
- **Capture instructor emphasis** — if Ramakrishna says "this is important" or "remember this", mark it with `> **Key Insight:**`

### Prohibited Patterns
- ❌ "The instructor explained X" → ✅ Write what X actually IS
- ❌ "Code shown in notebook" → ✅ Copy the code with comments
- ❌ Skipping breakout rooms → ✅ Document every student's design + feedback
- ❌ Skipping post-break content → ✅ The second half of sessions is equally important
- ❌ One-liner answers to student questions → ✅ Full explanation with context

---

## Standard Workflows

### Processing a New Session (5 Phases)

**Phase 0 — Inventory (2 min)**
```bash
ls -la "C:\Abhisss\IITM\AI_ML_Agentic\WeekN_Topic\DDthMon\"
```
List all files. Check: zip present? transcript present? already extracted?

**Phase 1 — Unzip (if needed)**
```bash
cd "C:\Abhisss\IITM\AI_ML_Agentic\WeekN_Topic\DDthMon"
unzip -o "16675 - Ramakrishna - *.zip" -d "DDthMon_extracted"
```

**Phase 2 — Transcript Survey (critical)**
- Read transcript in **500-line chunks** using `offset` parameter
- WEBVTT format: skip lines matching `\d{2}:\d{2}:\d{2}\.\d{3} --> \d{2}:\d{2}:\d{2}\.\d{3}`
- Read ALL chunks until end — a session is typically 5000–7000 lines total
- While reading, build a running outline: track topic shifts, Q&As, demos, breakouts

**Phase 3 — Code Cross-reference**
- Read all `.ipynb` and `.txt` files in `DDthMon_extracted/`
- Extract code patterns, function names, key implementations
- These fill in code sections that transcripts only describe verbally

**Phase 4 — Write Notes**
- Follow the template in `.claude/templates/notes-template.md`
- Write section by section, ensuring every transcript segment is covered
- Target: 400–600 lines for a standard 2.5hr session

**Phase 5 — Self-Audit**
- Re-read what you wrote
- Count sections — does it match the number of major topics in the transcript?
- Are all Q&As captured? Are all code snippets present? Are breakout rooms documented?

### Quick Git Commit
```bash
git -C "C:\Abhisss\IITM\AI_ML_Agentic" add "relative/path/to/file.md"
git -C "C:\Abhisss\IITM\AI_ML_Agentic" commit -m "Add WeekN DDthMon session notes"
git -C "C:\Abhisss\IITM\AI_ML_Agentic" push origin main
```

---

## Technical Conventions
- **Paths:** Always absolute in bash; use forward slashes; quote paths with spaces
- **Transcripts:** WEBVTT — timestamp format `HH:MM:SS.mmm --> HH:MM:SS.mmm` on its own line; skip these
- **Chunk reading:** `offset=1`, `offset=501`, `offset=1001`... with `limit=500` until EOF
- **Encoding:** UTF-8 for all notes files
- **Never stage:** .zip, .env, binary files, .ipynb unless explicitly asked

---

## Tech Stack Reference (for notes context)
| Category | Tools |
|----------|-------|
| Inference platforms | Groq (LPU-based, hosts Llama/others), OpenAI, Anthropic |
| LLMs used | Llama 3.1 8B Instant, GPT-4, Claude Opus |
| Agent frameworks | LangChain, LangGraph, CrewAI, AutoGen |
| Key APIs | PubMed/NCBI E-Utilities, Tavily, OpenFDA, RxNorm |
| UI tools | Gradio, Streamlit |
| Utilities | python-dotenv, requests, langchain-groq |

---

## Slash Commands Available
| Command | Purpose |
|---------|---------|
| `/new-session Week10/29thMar` | Full end-to-end session processing |
| `/push-notes` | Commit + push all new .md files |
| `/week-review Week10` | Consolidated study guide for a week |
| `/find-topic PubMed API` | Search all notes for a concept |
| `/deep-dive max_iterations` | Deep explanation of one concept |
| `/session-audit Week10/28thMar` | Audit existing notes for completeness |
| `/study-quiz Week10` | Generate quiz questions from notes |
| `/extract-code Week10` | Extract all code snippets to runnable files |
| `/concept-connect LangGraph` | Show connections across all weeks |
