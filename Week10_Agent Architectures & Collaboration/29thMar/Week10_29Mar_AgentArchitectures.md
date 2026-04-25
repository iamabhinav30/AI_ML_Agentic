# Week 10 – Agent Architectures & Collaboration
## Session: March 29, 2026 | 10:00 AM IST
**Presenter:** Abhinav Singh

---

## Session Overview

This session was a deep conceptual + practical class on:
- How to **design the right agent architecture** for a given problem
- **Reactive vs Deliberative vs Hybrid** agents
- **Single-agent vs Multi-agent** systems
- **Task-oriented vs Goal-oriented** agents
- **Communication patterns**: Sequential, Parallel, Hierarchical
- `max_iterations` — the hidden switch between task vs goal agents
- Introduction to **MCP (Model Context Protocol)**
- Hands-on code: Multi-agent system, CrewAI, goal-oriented flow notebooks

---

## 1. The Core Design Question

> "When you get a problem — what type of architecture do you need to design?"

Before writing any code, ask:

1. Do I need a simple LLM call, or a full agentic system?
2. Reactive or Deliberative agent?
3. Single-agent or Multi-agent?
4. Task-oriented or Goal-oriented?
5. How should agents communicate? Sequential / Parallel / Hierarchical?
6. Do I need a human in the loop?

There is **no one-size-fits-all answer** — it comes from experience and problem complexity.

---

## 2. Reactive vs Deliberative vs Hybrid Agents

### 2.1 Reactive Agents

| Attribute | Description |
|-----------|-------------|
| Thinking | None — instant response |
| Decision style | Rule-based: **If X → Do Y** |
| Internal model | No |
| History | No |
| Speed | Very fast |
| Use when | Simple, deterministic tasks |

**Real-world examples:**
- Thermostat (if temp > 30°C → turn on AC)
- Traffic light sensors
- High-frequency trading bots
- Simple FAQ chatbots
- Basic rule-based games

> **Key insight:** Rule-based `if-else` systems ARE agents — they just happen to be reactive ones.

---

### 2.2 Deliberative Agents

| Attribute | Description |
|-----------|-------------|
| Thinking | Deep — plans multiple steps ahead |
| Decision style | Simulates outcomes, then picks best action |
| Internal model | Yes — maintains world model |
| History | Yes |
| Speed | Slower (thinking takes time) |
| Use when | Complex, multi-step reasoning required |

**Real-world examples:**
- Chess AI (thinks 5+ moves ahead)
- Autonomous vehicles (route planning)
- Medical diagnosis systems
- Logistics optimization
- Strategic games

**Superpower of deliberative agents:** Handles complexity by simulating future states before acting.

**Disadvantage:** Slower, computationally expensive, can get stuck thinking (if iterations not bounded).

---

### 2.3 Hybrid Agents

- Combination of reactive + deliberative
- Some steps respond instantly; others require deep planning
- Most real-world production agents are hybrid
- Example: Customer support bot that instantly routes queries (reactive) but plans multi-step resolutions (deliberative)

---

## 3. Single-Agent vs Multi-Agent Systems

### When to use Single-Agent
- Simple, self-contained task
- One data source
- No parallel processing needed
- Example: "Summarize this document"

### When to use Multi-Agent
- Complex problem with **independent sub-tasks**
- Multiple data sources
- Specialization needed (each agent has a focused role)
- Speed through parallel execution
- Example: Insurance claim → Agent 1 fetches DB, Agent 2 reads policy PDF, Agent 3 compares, Agent 4 decides

### Multi-Agent Communication Patterns

#### Sequential
```
Agent 1 → Agent 2 → Agent 3 → Agent 4
```
- Output of one agent feeds the next
- Simple to implement
- **Drawback:** Introduces latency — each step waits for previous
- Use when: Each step genuinely depends on the previous result

#### Parallel
```
Agent 1 ──┐
Agent 2 ──┼──► Aggregator → Final Response
Agent 3 ──┘
```
- Multiple agents run simultaneously
- Results aggregated at the end
- Use when: Steps are **independent** of each other
- Example: Fetching data from 3 different websites — no reason to wait for one before calling others

#### Hierarchical
```
          [Orchestrator / Manager]
         /         |         \
    Agent 1    Agent 2    Agent 3
```
- A manager/orchestrator decides **which agents to call and when**
- CrewAI calls this component the **Crew Manager**
- Supports dynamic routing — manager decides at runtime
- Most flexible architecture

> **Ramakrishna's advice:** Ask yourself — "Why am I making these 3 website calls sequential? Are they dependent on each other?" If not, make them parallel. Sequential where not needed = unnecessary latency.

---

## 4. Task-Oriented vs Goal-Oriented Agents

### 4.1 Task-Oriented Agent

- Performs **one specific, pre-defined task**
- Single tool call, single output
- No multi-step reasoning
- Like: "Find dermatologist" → calls `find_doctors()` → returns list

**Example:**
```
Query: "Check doctor availability for tomorrow afternoon"
Action: Calls availability tool → Returns available/not available
Done. No further thinking.
```

---

### 4.2 Goal-Oriented Agent

- Works toward a **high-level goal** that may require multiple tasks
- Plans and executes **multiple steps** to achieve the goal
- Adapts based on intermediate results
- Like: "Book a neurology appointment tomorrow afternoon" → finds doctors → checks slots → if unavailable, suggests alternatives → confirms

**Example:**
```
Query: "Book a neurology appointment for tomorrow afternoon"

Step 1: Find all neurologists → [Dr. A, Dr. B, Dr. C]
Step 2: Check availability tomorrow afternoon → Dr. A: busy, Dr. B: available at 4pm
Step 3: Check if 4pm matches user preference → Yes (historical pattern: user books 4pm Sundays)
Step 4: Confirm booking
```

---

### 4.3 The `max_iterations` Switch — Critical Concept

> In LangChain's `AgentExecutor`, `max_iterations` is what makes an agent task-oriented vs goal-oriented.

```python
from langchain.agents import AgentExecutor

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    max_iterations=10,  # ← THIS IS THE KEY PARAMETER
    verbose=True
)
```

| `max_iterations` Value | Behavior |
|------------------------|----------|
| `1` | Task-oriented — agent thinks once, gives output, done |
| `5–10` | Goal-oriented — agent can plan, retry, and reason across steps |
| Very high (100+) | Agent keeps thinking indefinitely → high latency, runaway loops |

**Rule:** Always set `max_iterations` between **5 and 10** for goal-oriented agents.

> "Even if you write a beautiful goal-oriented prompt, if you set max_iterations=1, you have forced the agent to be task-oriented. The prompt doesn't override the iteration constraint."

### Debugging with max_iterations
If your agent isn't checking all data sources:
1. Check `max_iterations` — increase by 1 or 2
2. See if it now accesses the missing source
3. Don't always blame the prompt — sometimes the agent just wasn't given enough iterations to explore

---

## 5. How to Design an Architecture — Decision Framework

```
Start
  │
  ├─ Is this a simple, one-time task?
  │    └─ YES → Plain LLM call (no agent needed)
  │
  ├─ Does it require rules/triggers?
  │    └─ YES → Reactive Agent (rule-based)
  │
  ├─ Does it require planning + multi-step reasoning?
  │    └─ YES → Deliberative Agent
  │         │
  │         ├─ One task? → Task-Oriented (max_iterations=1)
  │         └─ Multi-step goal? → Goal-Oriented (max_iterations=5-10)
  │
  ├─ Multiple independent sub-tasks?
  │    └─ YES → Multi-Agent System
  │         │
  │         ├─ Steps depend on each other? → Sequential
  │         ├─ Steps are independent? → Parallel
  │         └─ Need dynamic routing? → Hierarchical (with Orchestrator)
  │
  └─ Human approval needed at any step?
       └─ YES → Human-in-the-Loop (HITL)
```

---

## 6. Human-in-the-Loop (HITL)

- Some workflows require human approval before the agent proceeds
- Example: Before sending an email, before making a financial transaction, before publishing a report
- Both LangGraph and CrewAI support HITL patterns
- The `Workflow creation+with_human_feedback.ipynb` in the extracted files covers this

---

## 7. MCP – Model Context Protocol (Preview)

> "Instead of statically wiring each agent to specific tools, expose ALL tools and let the agent decide which to use."

- Developed by **Anthropic**
- Standardizes how agents communicate with tools and frontend applications
- Eliminates the need to hardcode `tool_1`, `tool_2`, `tool_3` per agent
- All agents get access to a **shared tool registry** (100 tools available to all 4 agents)
- Agent dynamically selects the right tool at runtime

Also mentioned: **Agent-to-Agent Protocol** by Google — for agents communicating with other agents.

> "MCP is the future interface between agents and the software world. We'll study this in upcoming weeks."

---

## 8. Code & Notebooks Reference (from 29thMar_extracted/)

### Files Overview

| File | Content |
|------|---------|
| `practice.ipynb` | Basic multi-agent practice |
| `practice_fromRamakrishna.ipynb` | Instructor's reference implementation |
| `crewai.txt` | 7 domain practice problems |
| `Codes and data sets/multi_agent_system.ipynb` | Full multi-agent system |
| `Codes and data sets/crewai_multiagent.ipynb` | CrewAI multi-agent implementation |
| `Codes and data sets/goal_oriented_flow.ipynb` | Goal-oriented agent flow |
| `Codes and data sets/task_oriented_flow.ipynb` | Task-oriented agent flow |
| `Codes and data sets/Deliberative_agents_logistic optimize.ipynb` | Deliberative agent for logistics |
| `Codes and data sets/dynamic tool chain(optional).ipynb` | Dynamic tool selection |
| `Codes and data sets/Workflow creation+with_human_feedback.ipynb` | Human-in-the-loop workflow |
| `Codes and data sets/.env_example` | Environment variable template |

---

## 9. Real-World Engineering Mindset (from Ramakrishna)

- 95% of companies work at the **application layer** — calling LLM APIs, building tools, tuning prompts
- You won't be training models unless you're at OpenAI, Anthropic, or Google
- Your job: **Control prompts, tools, and agent iteration settings** to build a reliable application
- Tools often require **software engineering** (API integrations, web scraping, database connectors)
- Claude/Copilot can give you 60-70% of the code — the remaining 30% is your domain knowledge
- Always think: **can I parallelize this?** before making calls sequential

---

## 10. Key Takeaways

| Concept | Summary |
|---------|---------|
| Reactive agent | Rule-based, instant, no memory — if X then Y |
| Deliberative agent | Plans ahead, maintains world model, handles complexity |
| Hybrid | Mix of reactive + deliberative — most production systems |
| Task-oriented | Single step, `max_iterations=1` |
| Goal-oriented | Multi-step, `max_iterations=5-10` |
| Sequential | Each agent waits for the previous — only use when dependent |
| Parallel | Independent agents run simultaneously — reduces latency |
| Hierarchical | Orchestrator decides dynamically which agents to call |
| HITL | Human approval at critical decision points |
| MCP | Universal tool registry — agents discover and call tools dynamically |

---

## Files in this Folder

| File | Description |
|------|-------------|
| `16675 - Ramakrishna - 29th March 2026.zip` | All code and datasets for the session |
| `29thMar_extracted/` | Unzipped contents |
| `29thMar_extracted/Codes and data sets/` | All notebooks: multi-agent, CrewAI, goal-oriented, HITL, etc. |
| `Week 10_ Agent Architectures & Collaboration-1.pdf` | Session slides |
| `0 - 60614 - ... TRANSCRIPT.txt` | Full session transcript (WEBVTT format) |
