import streamlit as st
import warnings
warnings.filterwarnings("ignore")

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from crewai import Agent, Task

# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="Live AI Medical Triage",
    page_icon="🩺",
    layout="wide"
)

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ---------------- HEADER ----------------
st.title("🩺 Live AI Medical Triage System")
st.markdown("Observe each AI agent reasoning **step by step**.")

# ---------------- AGENTS ----------------
triage_agent = Agent(
    role="Triage Specialist",
    goal="Classify patient urgency.",
    backstory="Emergency triage expert trained on ESI and CTAS.",
    llm=llm,
    verbose=True
)

planning_agent = Agent(
    role="Care Planning Strategist",
    goal="Create treatment priorities.",
    backstory="Evidence-based medical planner.",
    llm=llm,
    verbose=True
)

decision_agent = Agent(
    role="Decision Intelligence Unit",
    goal="Choose emergency vs planned care.",
    backstory="Hybrid fast/slow medical reasoning engine.",
    llm=llm,
    verbose=True
)

monitoring_agent = Agent(
    role="Monitoring & Escalation Agent",
    goal="Detect deterioration and escalate.",
    backstory="Continuous patient safety observer.",
    llm=llm,
    verbose=True
)

# ---------------- INPUT ----------------
patient_details = st.text_area(
    "📝 Patient Symptoms & History",
    height=150,
    placeholder="Example: Severe chest pain, sweating, nausea, shortness of breath."
)

run = st.button("▶️ Run Live Assessment")

# ---------------- EXECUTION ----------------
if run and patient_details.strip():

    st.divider()
    st.subheader("🔄 Agent Execution Timeline")

    # ---- TRIAGE ----
    with st.spinner("🚑 Triage Specialist analyzing..."):
        triage_task = Task(
            description=f"""
Evaluate the following patient details:

{patient_details}

Steps:
1. Identify key symptoms
2. Classify severity (Mild / Moderate / Severe / Emergency)
3. Recommend next steps
""",
            expected_output="Structured triage report.",
            agent=triage_agent
        )
        triage_output = triage_task.execute()

    st.success("✅ Triage Completed")
    st.markdown("### 🚑 Triage Output")
    st.write(triage_output)

    # ---- CARE PLAN ----
    with st.spinner("🩹 Care Planning Strategist working..."):
        care_plan_task = Task(
            description=f"""
Using the triage findings and patient details:

{patient_details}

Generate a 3–5 step evidence-based care plan.
""",
            expected_output="Prioritized treatment plan.",
            agent=planning_agent
        )
        care_output = care_plan_task.execute()

    st.success("✅ Care Plan Generated")
    st.markdown("### 🩹 Care Plan")
    st.write(care_output)

    # ---- DECISION ----
    with st.spinner("🧠 Decision Intelligence reasoning..."):
        decision_task = Task(
            description=f"""
Based on the triage and care plan:

{patient_details}

Decide:
- Reactive Emergency Response
- Deliberative Planned Care

Provide justification.
""",
            expected_output="Decision with justification.",
            agent=decision_agent
        )
        decision_output = decision_task.execute()

    st.success("✅ Decision Made")
    st.markdown("### 🧠 Treatment Decision")
    st.write(decision_output)

    # ---- MONITORING ----
    with st.spinner("📡 Monitoring patient condition..."):
        monitoring_task = Task(
            description=f"""
Simulate ongoing monitoring for:

{patient_details}

If deterioration appears → ESCALATE IMMEDIATELY
Else → Patient stable
""",
            expected_output="Monitoring status.",
            agent=monitoring_agent
        )
        monitoring_output = monitoring_task.execute()

    if "ESCALATE" in monitoring_output.upper():
        st.error("🚨 ESCALATION REQUIRED")
    else:
        st.info("🟢 Patient Stable")

    st.markdown("### 📡 Monitoring Output")
    st.write(monitoring_output)

elif run:
    st.warning("Please enter patient details.")
