#!/usr/bin/env python
# coding: utf-8

"""
AI-Powered Customer Complaint Intelligence Platform — production release
-------------------------------------------------------------------------
RAG pipeline: OpenAI embeddings -> FAISS vector index -> GPT executive briefings.

Run:
  pip install -r requirements.txt
  streamlit run app.py

Secrets (.streamlit/secrets.toml):
  OPENAI_API_KEY = "sk-..."

Data: processed_complaints.csv in the app folder.
Required columns: theme, risk_score, recommendation, consumer_complaint_narrative
Optional columns (auto-detected, enable extra views): company, state,
date_received / date received / date / complaint_date
"""

import os
import hashlib
from datetime import datetime

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="Complaint Intelligence",
    page_icon=":shield:",
    layout="wide",
)

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4o-mini"
CACHE_DIR = ".embed_cache"
MAX_ROWS = 3000

# -----------------------------
# Design system
# -----------------------------
INK = "#16233B"        # deep navy — primary text & structure
INK_SOFT = "#5B6B85"   # muted slate — secondary text
PANEL = "#F4F6FA"      # cool panel background
LINE = "#DDE3EE"
AMBER = "#B45309"      # risk accent
RED = "#B42318"
GREEN = "#067647"

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@500;700&family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; color: {INK}; }}

.main-title {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 40px; font-weight: 700; color: {INK};
    letter-spacing: -0.5px; margin-bottom: 2px;
}}
.subtitle {{ font-size: 15px; color: {INK_SOFT}; margin-bottom: 8px; }}

.section-header {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 24px; font-weight: 700; color: {INK};
    margin-top: 12px; margin-bottom: 12px;
}}

.kpi-card {{
    background-color: {PANEL}; border: 1px solid {LINE}; border-radius: 14px;
    padding: 18px 20px; min-height: 128px;
}}
.kpi-label {{
    font-size: 12px; color: {INK_SOFT}; font-weight: 600;
    text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 10px;
}}
.kpi-value {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 26px; color: {INK}; font-weight: 700; line-height: 1.25;
    white-space: normal; overflow-wrap: break-word;
}}

.agent-hero {{
    background: {INK}; color: #F5F7FB; border-radius: 16px;
    padding: 28px 32px; margin-bottom: 18px;
}}
.agent-hero h2 {{
    font-family: 'Space Grotesk', sans-serif;
    font-size: 28px; font-weight: 700; color: #FFFFFF;
    margin: 0 0 6px 0; letter-spacing: -0.3px;
}}
.agent-hero p {{ font-size: 14.5px; color: #C4CEDF; margin: 0; max-width: 720px; }}
.agent-hero .pipeline {{
    margin-top: 14px; font-size: 12.5px; color: #8FA0BC;
    font-weight: 600; letter-spacing: 0.04em;
}}
.agent-hero .pipeline b {{ color: #E9C46A; font-weight: 700; }}

.badge {{
    display: inline-block; border-radius: 999px; padding: 3px 12px;
    font-size: 12px; font-weight: 700;
}}
.badge-high {{ background: #FEE4E2; color: {RED}; }}
.badge-mod  {{ background: #FEF0C7; color: {AMBER}; }}
.badge-low  {{ background: #D1FADF; color: {GREEN}; }}
.cite-chip {{
    display: inline-block; background: {PANEL}; color: {INK};
    border: 1px solid {LINE}; border-radius: 8px; padding: 2px 8px;
    font-size: 12px; font-weight: 600; margin: 2px;
}}

.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 1px solid {LINE}; }}
.stTabs [data-baseweb="tab"] {{ font-weight: 600; color: {INK_SOFT}; padding: 10px 16px; }}
.stTabs [aria-selected="true"] {{
    color: {INK} !important; border-bottom: 3px solid {AMBER} !important;
}}

.stButton > button[kind="primary"] {{
    background: {INK}; border: 1px solid {INK}; color: #fff; font-weight: 600;
}}
.stButton > button[kind="primary"]:hover {{ background: #223354; border-color: #223354; }}

section[data-testid="stSidebar"] {{ background: {PANEL}; border-right: 1px solid {LINE}; }}
</style>
""",
    unsafe_allow_html=True,
)


def style_fig(fig, height=520):
    fig.update_layout(
        height=height,
        font=dict(family="Inter, sans-serif", color=INK, size=13),
        title_font=dict(family="Space Grotesk, sans-serif", size=17, color=INK),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=10),
        yaxis=dict(gridcolor=LINE),
        xaxis=dict(gridcolor=LINE),
    )
    return fig


def risk_badge(level):
    cls = {"High": "badge-high", "Moderate": "badge-mod", "Low": "badge-low"}[level]
    return f'<span class="badge {cls}">{level} risk</span>'


def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------
# API key
# -----------------------------
def get_api_key():
    try:
        return st.secrets["OPENAI_API_KEY"]
    except Exception:
        return os.environ.get("OPENAI_API_KEY")


API_KEY = get_api_key()


# -----------------------------
# Load data (with optional-column detection)
# -----------------------------
DATE_CANDIDATES = ["date_received", "date received", "complaint_date", "date"]


@st.cache_data
def load_data():
    df = pd.read_csv("processed_complaints.csv")
    df.columns = df.columns.str.strip().str.lower()

    required_cols = [
        "theme",
        "risk_score",
        "recommendation",
        "consumer_complaint_narrative",
    ]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        st.error(
            f"Missing required columns in CSV: {missing}. "
            "Expected: theme, risk_score, recommendation, consumer_complaint_narrative."
        )
        st.stop()

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].astype(str)
    df["theme"] = df["theme"].astype(str)
    df["recommendation"] = df["recommendation"].astype(str)
    df = df.dropna(subset=required_cols).reset_index(drop=True)

    if "company" in df.columns:
        df["company"] = df["company"].astype(str)
    if "state" in df.columns:
        df["state"] = df["state"].astype(str).str.strip().str.upper()

    # Detect and parse a date column if present.
    date_col = next((c for c in DATE_CANDIDATES if c in df.columns), None)
    if date_col:
        df["_date"] = pd.to_datetime(df[date_col], errors="coerce")
        if df["_date"].notna().sum() == 0:
            df = df.drop(columns=["_date"])

    df = df.sample(n=min(MAX_ROWS, len(df)), random_state=42).reset_index(drop=True)
    return df


pdf = load_data()
HAS_COMPANY = "company" in pdf.columns
HAS_STATE = "state" in pdf.columns
HAS_DATE = "_date" in pdf.columns


# -----------------------------
# Embeddings + FAISS (real RAG)
# -----------------------------
def _corpus_fingerprint(texts):
    h = hashlib.sha256()
    h.update(str(len(texts)).encode())
    for t in texts[:50]:
        h.update(t[:200].encode("utf-8", "ignore"))
    return h.hexdigest()[:16]


def _embed_batch(client, texts):
    vectors = []
    BATCH = 256
    for i in range(0, len(texts), BATCH):
        chunk = [t[:6000] for t in texts[i : i + BATCH]]
        resp = client.embeddings.create(model=EMBED_MODEL, input=chunk)
        vectors.extend([d.embedding for d in resp.data])
    return np.array(vectors, dtype="float32")


@st.cache_resource(show_spinner="Building semantic index (embeddings + FAISS)...")
def build_faiss_index(texts, fingerprint):
    import faiss
    from openai import OpenAI

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"emb_{fingerprint}.npy")

    if os.path.exists(cache_path):
        embeddings = np.load(cache_path)
    else:
        client = OpenAI(api_key=API_KEY)
        embeddings = _embed_batch(client, texts)
        np.save(cache_path, embeddings)

    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index


def embed_query(query):
    import faiss
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY)
    resp = client.embeddings.create(model=EMBED_MODEL, input=[query[:6000]])
    vec = np.array([resp.data[0].embedding], dtype="float32")
    faiss.normalize_L2(vec)
    return vec


FAISS_INDEX = None
INDEX_ERROR = None
if API_KEY:
    try:
        _texts = pdf["consumer_complaint_narrative"].tolist()
        _fp = _corpus_fingerprint(_texts)
        FAISS_INDEX = build_faiss_index(_texts, _fp)
    except Exception as e:
        INDEX_ERROR = str(e)


# -----------------------------
# Retrieval + LLM
# -----------------------------
def retrieve_relevant_complaints(query, data, k=10):
    """Semantic retrieval over FAISS, restricted to the currently filtered rows."""
    if len(data) == 0 or FAISS_INDEX is None:
        out = data.copy()
        out["semantic_score"] = []
        return out

    allowed_positions = pdf.index.get_indexer(data.index)
    allowed_set = set(int(p) for p in allowed_positions if p != -1)

    qvec = embed_query(query)
    search_k = min(len(pdf), max(k * 10, 50))
    scores, ids = FAISS_INDEX.search(qvec, search_k)

    kept = []
    for score, idx in zip(scores[0], ids[0]):
        if idx in allowed_set:
            kept.append((idx, float(score)))
        if len(kept) >= k:
            break

    if not kept:
        out = data.head(k).copy()
        out["semantic_score"] = 0.0
        return out

    pos_list = [idx for idx, _ in kept]
    result = pdf.iloc[pos_list].copy()
    result["semantic_score"] = [s for _, s in kept]
    return result


def answer_business_question(query, data, k=10):
    results = retrieve_relevant_complaints(query, data, k)
    theme_counts = results["theme"].value_counts()
    avg_risk = results["risk_score"].mean() if len(results) else 0.0
    risk_level = "High" if avg_risk >= 8 else "Moderate" if avg_risk >= 5 else "Low"
    recommendations = results["recommendation"].dropna().unique().tolist()
    return results, theme_counts, avg_risk, risk_level, recommendations


def generate_llm_answer(question, results):
    from openai import OpenAI

    client = OpenAI(api_key=API_KEY)

    context = ""
    for i, (_, row) in enumerate(results.head(10).iterrows(), start=1):
        company = row["company"] if HAS_COMPANY else "N/A"
        context += (
            f"[Source {i}] Company: {company} | Theme: {row['theme']} | "
            f"Risk: {round(row['risk_score'], 2)} | "
            f"Recommendation: {row['recommendation']}\n"
            f"Complaint: {row['consumer_complaint_narrative'][:1000]}\n---\n"
        )

    prompt = f"""You are a senior business analytics consultant.

Use ONLY the complaint evidence below to answer the user's business question.
When you make a claim, cite the source number in square brackets, e.g. [Source 3].

User Question:
{question}

Complaint Evidence:
{context}

Return your answer in this format:

### Executive Summary
### Key Complaint Drivers
### Risk Assessment
### Recommended Business Actions
### Supporting Evidence

Be specific to the user's question. Avoid generic summaries."""

    resp = client.chat.completions.create(
        model=CHAT_MODEL,
        messages=[
            {
                "role": "system",
                "content": "You are an expert in customer analytics, financial "
                "services operations, risk, and customer experience.",
            },
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content


def friendly_openai_error(e):
    """Translate OpenAI/network exceptions into actionable user-facing messages."""
    msg = str(e)
    if "401" in msg or "authentication" in msg.lower() or "invalid_api_key" in msg.lower():
        return ("Your OpenAI API key was rejected. Check the key in "
                "`.streamlit/secrets.toml` and restart the app.")
    if "429" in msg or "rate" in msg.lower():
        return ("OpenAI rate limit or quota reached. Wait a moment and try again, "
                "or check your usage limits at platform.openai.com.")
    if "timeout" in msg.lower() or "connection" in msg.lower():
        return "Could not reach OpenAI. Check your internet connection and try again."
    return f"The AI request failed: {msg[:200]}"


# -----------------------------
# Session state
# -----------------------------
if "briefing_history" not in st.session_state:
    st.session_state.briefing_history = []      # list of dicts
if "answer_cache" not in st.session_state:
    st.session_state.answer_cache = {}          # (question, ids) -> answer


# -----------------------------
# Header
# -----------------------------
st.markdown('<div class="main-title">Complaint Intelligence Platform</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">CFPB consumer complaints &middot; NLP risk scoring &middot; '
    "RAG-powered executive answers</div>",
    unsafe_allow_html=True,
)

with st.expander("About this platform"):
    st.markdown(
        f"""
**Data** — CFPB Consumer Complaint Database (public), preprocessed offline with
topic modeling and risk scoring into `processed_complaints.csv`
({len(pdf):,} complaints sampled for interactive performance).

**Architecture** — Complaint narratives are embedded once with `{EMBED_MODEL}`
and indexed in FAISS (cosine similarity, disk-cached). At question time the
query is embedded, the top matches are retrieved, and `{CHAT_MODEL}` produces a
briefing grounded only in that evidence, with `[Source N]` citations.

**Cost controls** — Embeddings are cached to disk; briefings are cached per
question within a session so reruns never re-bill.
"""
    )

if not API_KEY:
    st.error(
        "**OpenAI API key not found.** The AI agent and semantic search are disabled.\n\n"
        "Add your key to `.streamlit/secrets.toml`:\n\n"
        "```toml\nOPENAI_API_KEY = \"sk-...\"\n```\n\n"
        "or set the `OPENAI_API_KEY` environment variable, then restart the app. "
        "The dashboard tabs still work without a key."
    )
elif INDEX_ERROR:
    st.error(f"Could not build the semantic index: {friendly_openai_error(INDEX_ERROR)}")


# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters")

company_options = ["All"] + (sorted(pdf["company"].dropna().unique().tolist()) if HAS_COMPANY else [])
company_filter = st.sidebar.selectbox("Company", company_options)
theme_filter = st.sidebar.selectbox("Theme", ["All"] + sorted(pdf["theme"].dropna().unique().tolist()))

filtered = pdf.copy()
if company_filter != "All" and HAS_COMPANY:
    filtered = filtered[filtered["company"] == company_filter]
if theme_filter != "All":
    filtered = filtered[filtered["theme"] == theme_filter]

if HAS_DATE:
    valid_dates = filtered["_date"].dropna()
    if len(valid_dates) > 1:
        dmin, dmax = valid_dates.min().date(), valid_dates.max().date()
        if dmin < dmax:
            date_range = st.sidebar.slider(
                "Date range", min_value=dmin, max_value=dmax, value=(dmin, dmax)
            )
            filtered = filtered[
                filtered["_date"].isna()
                | (
                    (filtered["_date"].dt.date >= date_range[0])
                    & (filtered["_date"].dt.date <= date_range[1])
                )
            ]

st.sidebar.caption(f"{len(filtered):,} complaints in scope")

if st.sidebar.button("Reset filters"):
    st.session_state.clear()
    st.rerun()


# -----------------------------
# KPI row
# -----------------------------
if len(filtered) > 0:
    highest_risk_theme = filtered.groupby("theme")["risk_score"].mean().idxmax()
    avg_risk_score = round(filtered["risk_score"].mean(), 2)
    themes_discovered = filtered["theme"].nunique()
else:
    highest_risk_theme = "N/A"
    avg_risk_score = "N/A"
    themes_discovered = 0

col1, col2, col3, col4 = st.columns([1.1, 1.1, 1.1, 1.7])
with col1:
    kpi_card("Complaints Analyzed", f"{len(filtered):,}")
with col2:
    kpi_card("Themes Discovered", themes_discovered)
with col3:
    kpi_card("Avg Risk Score", avg_risk_score)
with col4:
    kpi_card("Highest Risk Theme", highest_risk_theme)

st.divider()


# -----------------------------
# Tabs
# -----------------------------
tab_agent, tab_exec, tab_theme, tab_risk = st.tabs(
    ["AI Analyst", "Executive Dashboard", "Theme Analysis", "High-Risk Queue"]
)

SUGGESTED_QUESTIONS = [
    "What fraud patterns are emerging?",
    "Which billing issues drive the most risk?",
    "What are customers saying about account closures?",
    "Where should leadership focus remediation first?",
]

if "agent_question" not in st.session_state:
    st.session_state.agent_question = SUGGESTED_QUESTIONS[0]


def _set_question(q):
    st.session_state.agent_question = q


def render_briefing(entry):
    """Render one briefing (current or from history)."""
    st.markdown(
        f"### Executive Briefing &nbsp; {risk_badge(entry['risk_level'])}",
        unsafe_allow_html=True,
    )
    st.markdown(entry["answer"])

    if entry.get("source_chips"):
        st.markdown("**Grounded in:** " + entry["source_chips"], unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        kpi_card("Average Retrieved Risk Score", round(entry["avg_risk"], 2))
    with c2:
        kpi_card("Risk Level", entry["risk_level"])

    st.download_button(
        "Download briefing (Markdown)",
        data=(
            f"# Executive Briefing\n\n**Question:** {entry['question']}\n\n"
            f"**Risk level:** {entry['risk_level']} "
            f"(avg retrieved risk {round(entry['avg_risk'], 2)})\n\n"
            f"**Generated:** {entry['timestamp']}\n\n---\n\n{entry['answer']}\n"
        ),
        file_name="executive_briefing.md",
        mime="text/markdown",
        key=f"dl_{entry['timestamp']}",
    )


# -----------------------------
# Tab 1 — AI Agent
# -----------------------------
with tab_agent:
    st.markdown(
        """
        <div class="agent-hero">
            <h2>Ask the Complaint Intelligence Analyst</h2>
            <p>Ask any business question in plain English. The AI Analyst retrieves the most
            relevant complaints from the database and produces a cited executive briefing
            grounded only in that evidence &mdash; no generic answers.</p>
            <div class="pipeline">PIPELINE &nbsp;&middot;&nbsp; Question
            &rarr; <b>OpenAI embedding</b> &rarr; <b>FAISS retrieval</b>
            &rarr; <b>Briefing with citations</b></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.caption("Try one of these:")
    chip_cols = st.columns(len(SUGGESTED_QUESTIONS))
    for col, q in zip(chip_cols, SUGGESTED_QUESTIONS):
        col.button(q, key=f"chip_{q}", on_click=_set_question, args=(q,), use_container_width=True)

    question = st.text_area(
        "Business question", key="agent_question", height=100, label_visibility="collapsed"
    )

    left, right = st.columns([3, 1])
    with right:
        k = st.slider("Complaints to retrieve", 5, 25, 10)
    with left:
        run = st.button("Generate Executive Briefing", type="primary", use_container_width=True)

    if run:
        if not API_KEY or FAISS_INDEX is None:
            st.error("The AI agent needs a working OpenAI API key (see message at top).")
        elif not question.strip():
            st.warning("Type a business question first, or pick one of the suggestions above.")
        elif len(filtered) == 0:
            st.warning("No complaints match the current filters. Widen the filters in the sidebar.")
        else:
            try:
                with st.status("Working on your briefing...", expanded=True) as status:
                    st.write("Retrieving the most relevant complaints...")
                    results, theme_counts, avg_risk, risk_level, recommendations = (
                        answer_business_question(question, filtered, k)
                    )

                    cache_key = (question.strip().lower(), tuple(results.index.tolist()))
                    if cache_key in st.session_state.answer_cache:
                        st.write("Found a cached briefing for this exact question and evidence.")
                        answer = st.session_state.answer_cache[cache_key]
                    else:
                        st.write(f"Asking your AI Analyst for a cited executive briefing...")
                        answer = generate_llm_answer(question, results)
                        st.session_state.answer_cache[cache_key] = answer

                    status.update(label="Briefing ready", state="complete", expanded=False)

                source_chips = ""
                if HAS_COMPANY:
                    source_chips = "".join(
                        f'<span class="cite-chip">Source {i}: {row["company"][:30]}</span>'
                        for i, (_, row) in enumerate(results.head(10).iterrows(), start=1)
                    )

                entry = {
                    "question": question,
                    "answer": answer,
                    "avg_risk": float(avg_risk),
                    "risk_level": risk_level,
                    "source_chips": source_chips,
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                }
                st.session_state.briefing_history.insert(0, entry)

                render_briefing(entry)

                st.subheader("Key Complaint Drivers")
                driver_df = theme_counts.reset_index()
                driver_df.columns = ["Theme", "Retrieved Complaints"]
                st.dataframe(driver_df, use_container_width=True)

                st.subheader("Recommended Business Actions")
                for rec in recommendations[:5]:
                    st.markdown(f"- {rec}")

                st.subheader("Supporting Complaint Evidence")
                for i, (_, row) in enumerate(results.head(5).iterrows(), start=1):
                    score = row.get("semantic_score", 0)
                    with st.expander(
                        f"[Source {i}] {row['theme']} | Risk: {round(row['risk_score'], 2)} "
                        f"| Similarity: {round(score, 3)}"
                    ):
                        st.write(row["consumer_complaint_narrative"][:1500])

            except Exception as e:
                st.error(friendly_openai_error(e))

    # Briefing history
    past = st.session_state.briefing_history[1:] if run else st.session_state.briefing_history
    if past:
        st.divider()
        st.subheader("Previous briefings this session")
        for entry in past[:5]:
            with st.expander(f"{entry['timestamp']} — {entry['question'][:80]}"):
                render_briefing(entry)

# -----------------------------
# Tab 2 — Executive Dashboard
# -----------------------------
with tab_exec:
    st.markdown('<div class="section-header">Executive Overview</div>', unsafe_allow_html=True)

    if len(filtered) == 0:
        st.warning("No complaints match the current filters. Widen the filters in the sidebar.")
    else:
        left, right = st.columns(2)

        theme_volume = filtered["theme"].value_counts().reset_index()
        theme_volume.columns = ["Theme", "Complaint Count"]
        fig1 = px.bar(
            theme_volume, x="Complaint Count", y="Theme", orientation="h",
            title="Complaint Volume by Theme", color_discrete_sequence=[INK],
        )
        fig1.update_layout(yaxis={"categoryorder": "total ascending"})
        left.plotly_chart(style_fig(fig1, 520), use_container_width=True)

        theme_risk = (
            filtered.groupby("theme")
            .agg(avg_risk_score=("risk_score", "mean"))
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )
        fig2 = px.bar(
            theme_risk, x="avg_risk_score", y="theme", orientation="h",
            title="Average Risk Score by Theme", color_discrete_sequence=[AMBER],
        )
        fig2.update_layout(yaxis={"categoryorder": "total ascending"})
        right.plotly_chart(style_fig(fig2, 520), use_container_width=True)

        # Trend over time (if date column present)
        if HAS_DATE and filtered["_date"].notna().sum() > 10:
            monthly = (
                filtered.dropna(subset=["_date"])
                .set_index("_date")
                .resample("MS")
                .agg(complaints=("theme", "size"), avg_risk=("risk_score", "mean"))
                .reset_index()
            )
            if len(monthly) > 1:
                fig3 = px.line(
                    monthly, x="_date", y="complaints",
                    title="Complaint Volume Over Time (monthly)",
                    color_discrete_sequence=[INK], markers=True,
                )
                fig3.update_xaxes(title="")
                st.plotly_chart(style_fig(fig3, 360), use_container_width=True)

        # Geography (if state column present)
        if HAS_STATE:
            state_agg = (
                filtered.groupby("state")
                .agg(complaints=("theme", "size"), avg_risk=("risk_score", "mean"))
                .reset_index()
            )
            state_agg = state_agg[state_agg["state"].str.len() == 2]
            if len(state_agg) > 3:
                fig4 = px.choropleth(
                    state_agg,
                    locations="state",
                    locationmode="USA-states",
                    color="avg_risk",
                    scope="usa",
                    hover_data={"complaints": True, "avg_risk": ":.2f"},
                    color_continuous_scale=["#D1FADF", "#FEF0C7", "#FEE4E2", RED],
                    title="Average Risk Score by State",
                )
                fig4.update_layout(geo=dict(bgcolor="rgba(0,0,0,0)"))
                st.plotly_chart(style_fig(fig4, 480), use_container_width=True)

        # Company risk leaderboard (if company column present)
        if HAS_COMPANY and company_filter == "All":
            st.subheader("Company Risk Leaderboard")
            leaderboard = (
                filtered.groupby("company")
                .agg(complaints=("theme", "size"), avg_risk=("risk_score", "mean"))
                .reset_index()
                .query("complaints >= 5")
                .sort_values(["avg_risk", "complaints"], ascending=False)
                .head(10)
            )
            if len(leaderboard) > 0:
                leaderboard["avg_risk"] = leaderboard["avg_risk"].round(2)
                leaderboard.columns = ["Company", "Complaints", "Avg Risk"]
                st.dataframe(leaderboard, use_container_width=True, hide_index=True)
            else:
                st.caption("Not enough complaints per company to rank (minimum 5).")

        st.subheader("Recommendation Summary")
        rec_summary = (
            filtered.groupby(["theme", "recommendation"])
            .size()
            .reset_index(name="complaint_count")
            .sort_values("complaint_count", ascending=False)
        )
        st.dataframe(rec_summary, use_container_width=True)

# -----------------------------
# Tab 3 — Theme Analysis
# -----------------------------
with tab_theme:
    st.markdown('<div class="section-header">Explore Complaint Themes</div>', unsafe_allow_html=True)

    if len(filtered) == 0:
        st.warning("No complaints match the current filters. Widen the filters in the sidebar.")
    else:
        selected_theme = st.selectbox(
            "Select a theme to explore", sorted(filtered["theme"].dropna().unique())
        )
        theme_df = filtered[filtered["theme"] == selected_theme]

        c1, c2, c3 = st.columns(3)
        with c1:
            kpi_card("Complaints", f"{len(theme_df):,}")
        with c2:
            kpi_card("Avg Risk", round(theme_df["risk_score"].mean(), 2))
        with c3:
            kpi_card("Max Risk", round(theme_df["risk_score"].max(), 2))

        st.markdown("### Recommended Action")
        st.info(theme_df["recommendation"].mode().iloc[0])

        st.markdown("### Sample Complaints")
        for _, row in theme_df.sort_values("risk_score", ascending=False).head(5).iterrows():
            with st.expander(f"Risk Score: {round(row['risk_score'], 2)}"):
                st.write(row["consumer_complaint_narrative"][:1200])

# -----------------------------
# Tab 4 — High-Risk Queue
# -----------------------------
with tab_risk:
    st.markdown('<div class="section-header">High-Risk Complaint Queue</div>', unsafe_allow_html=True)

    if len(filtered) == 0:
        st.warning("No complaints match the current filters. Widen the filters in the sidebar.")
    else:
        c1, c2 = st.columns([1, 1])
        with c1:
            risk_threshold = st.slider(
                "Minimum Risk Score",
                float(filtered["risk_score"].min()),
                float(filtered["risk_score"].max()),
                float(filtered["risk_score"].quantile(0.90)),
            )
        with c2:
            keyword = st.text_input("Search within complaints", placeholder="e.g. unauthorized charge")

        high_risk = (
            filtered[filtered["risk_score"] >= risk_threshold]
            .sort_values("risk_score", ascending=False)
        )
        if keyword.strip():
            high_risk = high_risk[
                high_risk["consumer_complaint_narrative"].str.contains(
                    keyword.strip(), case=False, na=False
                )
            ]

        st.write(f"{len(high_risk):,} complaints match.")

        cols_to_show = ["theme", "risk_score", "recommendation", "consumer_complaint_narrative"]
        if HAS_COMPANY:
            cols_to_show = ["company"] + cols_to_show
        st.dataframe(high_risk[cols_to_show], use_container_width=True)

        st.download_button(
            "Download queue (CSV)",
            data=high_risk[cols_to_show].to_csv(index=False).encode("utf-8"),
            file_name="high_risk_complaints.csv",
            mime="text/csv",
        )

st.caption(
    "Complaint Intelligence Platform · CFPB public data · "
    "Embeddings and briefings by OpenAI · Retrieval by FAISS"
)
