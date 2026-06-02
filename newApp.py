#!/usr/bin/env python
# coding: utf-8

# In[ ]:
 #!/usr/bin/env python
# coding: utf-8

#!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="AI Complaint Intelligence Platform",
    layout="wide"
)

# -----------------------------
# Custom Styling
# -----------------------------
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    .main-title {
        font-size: 42px;
        font-weight: 800;
        color: #111827;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 15px;
        color: #6B7280;
        margin-bottom: 30px;
    }

    .kpi-card {
        background-color: #F8FAFC;
        border: 1px solid #E5E7EB;
        border-radius: 16px;
        padding: 20px;
        min-height: 140px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .kpi-label {
        font-size: 13px;
        color: #6B7280;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        margin-bottom: 12px;
    }

    .kpi-value {
        font-size: 26px;
        color: #111827;
        font-weight: 800;
        line-height: 1.25;
        white-space: normal;
        overflow-wrap: break-word;
        word-break: normal;
    }

    .section-header {
        font-size: 26px;
        font-weight: 800;
        color: #111827;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .small-note {
        font-size: 14px;
        color: #6B7280;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -----------------------------
# Load Data
# -----------------------------
@st.cache_data
def load_data():
    df = pd.read_csv("processed_complaints.csv")
    df.columns = df.columns.str.strip().str.lower()

    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].astype(str)

    required_cols = [
        "theme",
        "risk_score",
        "recommendation",
        "consumer_complaint_narrative"
    ]

    df = df.dropna(subset=required_cols)

    if "company" in df.columns:
        df["company"] = df["company"].astype(str)

    return df


pdf = load_data()

# -----------------------------
# Helper Functions
# -----------------------------
def kpi_card(title, value):
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="kpi-label">{title}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True
    )


def retrieve_relevant_complaints(query, data, k=10):
    query_words = query.lower().split()

    temp = data.copy()

    temp["match_score"] = temp["consumer_complaint_narrative"].astype(str).apply(
        lambda x: sum(word in x.lower() for word in query_words)
    )

    results = (
        temp.sort_values(
            ["match_score", "risk_score"],
            ascending=False
        )
        .head(k)
    )

    return results


def answer_business_question(query, data, k=10):
    results = retrieve_relevant_complaints(query, data, k)

    theme_counts = results["theme"].value_counts()
    avg_risk = results["risk_score"].mean()

    risk_level = (
        "High" if avg_risk >= 8
        else "Moderate" if avg_risk >= 5
        else "Low"
    )

    recommendations = results["recommendation"].dropna().unique().tolist()
    top_themes = theme_counts.head(3).index.tolist()

    summary = (
        f"Based on the retrieved complaints, the most relevant customer pain points are "
        f"{', '.join(top_themes)}. The average retrieved risk score is {avg_risk:.2f}, "
        f"indicating a {risk_level.lower()} level of business risk."
    )

    return results, summary, theme_counts, avg_risk, risk_level, recommendations


# -----------------------------
# Header
# -----------------------------
st.markdown(
    '<div class="main-title">AI-Powered Customer Complaint Intelligence Platform</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">NLP + Risk Scoring + Semantic Retrieval + Executive Recommendations</div>',
    unsafe_allow_html=True
)

# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

if "company" in pdf.columns:
    company_options = ["All"] + sorted(
        pdf["company"].dropna().astype(str).unique().tolist()
    )
else:
    company_options = ["All"]

company_filter = st.sidebar.selectbox(
    "Company",
    company_options
)

theme_filter = st.sidebar.selectbox(
    "Theme",
    ["All"] + sorted(pdf["theme"].dropna().unique().tolist())
)

filtered = pdf.copy()

if company_filter != "All" and "company" in filtered.columns:
    filtered = filtered[filtered["company"] == company_filter]

if theme_filter != "All":
    filtered = filtered[filtered["theme"] == theme_filter]

# -----------------------------
# KPI Cards
# -----------------------------
if len(filtered) > 0:
    highest_risk_theme = (
        filtered.groupby("theme")["risk_score"]
        .mean()
        .idxmax()
    )

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
tab1, tab2, tab3, tab4 = st.tabs(
    [
        "Executive Dashboard",
        "Theme Analysis",
        "AI Agent",
        "High-Risk Complaints"
    ]
)

# -----------------------------
# Tab 1: Executive Dashboard
# -----------------------------
with tab1:
    st.markdown(
        '<div class="section-header">Executive Overview</div>',
        unsafe_allow_html=True
    )

    if len(filtered) == 0:
        st.warning("No data available for the selected filters.")
    else:
        left, right = st.columns(2)

        theme_volume = (
            filtered["theme"]
            .value_counts()
            .reset_index()
        )

        theme_volume.columns = [
            "Theme",
            "Complaint Count"
        ]

        fig1 = px.bar(
            theme_volume,
            x="Complaint Count",
            y="Theme",
            orientation="h",
            title="Complaint Volume by Theme"
        )

        fig1.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=560,
            font=dict(size=12),
            title_font=dict(size=18)
        )

        left.plotly_chart(
            fig1,
            use_container_width=True
        )

        theme_risk = (
            filtered.groupby("theme")
            .agg(avg_risk_score=("risk_score", "mean"))
            .reset_index()
            .sort_values("avg_risk_score", ascending=False)
        )

        fig2 = px.bar(
            theme_risk,
            x="avg_risk_score",
            y="theme",
            orientation="h",
            title="Average Risk Score by Theme"
        )

        fig2.update_layout(
            yaxis={"categoryorder": "total ascending"},
            height=560,
            font=dict(size=12),
            title_font=dict(size=18)
        )

        right.plotly_chart(
            fig2,
            use_container_width=True
        )

        st.subheader("Recommendation Summary")

        rec_summary = (
            filtered.groupby(["theme", "recommendation"])
            .size()
            .reset_index(name="complaint_count")
            .sort_values("complaint_count", ascending=False)
        )

        st.dataframe(
            rec_summary,
            use_container_width=True
        )

# -----------------------------
# Tab 2: Theme Analysis
# -----------------------------
with tab2:
    st.markdown(
        '<div class="section-header">Explore Complaint Themes</div>',
        unsafe_allow_html=True
    )

    if len(filtered) == 0:
        st.warning("No data available for the selected filters.")
    else:
        selected_theme = st.selectbox(
            "Select a theme to explore",
            sorted(filtered["theme"].dropna().unique())
        )

        theme_df = filtered[
            filtered["theme"] == selected_theme
        ]

        c1, c2, c3 = st.columns(3)

        with c1:
            kpi_card("Complaints", f"{len(theme_df):,}")

        with c2:
            kpi_card("Avg Risk", round(theme_df["risk_score"].mean(), 2))

        with c3:
            kpi_card("Max Risk", round(theme_df["risk_score"].max(), 2))

        st.markdown("### Recommended Action")

        if len(theme_df) > 0:
            st.info(
                theme_df["recommendation"]
                .mode()
                .iloc[0]
            )

        st.markdown("### Sample Complaints")

        for _, row in theme_df.head(5).iterrows():
            with st.expander(
                f"Risk Score: {round(row['risk_score'], 2)}"
            ):
                st.write(
                    row["consumer_complaint_narrative"][:1200]
                )

# -----------------------------
# Tab 3: AI Agent
# -----------------------------
with tab3:
    st.markdown(
        '<div class="section-header">AI Complaint Intelligence Agent</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        Ask a business question and the system will retrieve relevant complaints, 
        summarize key drivers, assess business risk, and recommend actions.
        """
    )

    question = st.text_area(
        "Ask a business question",
        value="What are the top fraud-related issues and what should leadership do?",
        height=120
    )

    k = st.slider(
        "Number of complaints to retrieve",
        min_value=5,
        max_value=25,
        value=10
    )

    if st.button(
        "Generate Executive Summary"
    ):
        if len(filtered) == 0:
            st.warning("No data available for the selected filters.")
        else:
            (
                results,
                summary,
                theme_counts,
                avg_risk,
                risk_level,
                recommendations
            ) = answer_business_question(
                question,
                filtered,
                k
            )

            st.subheader("Executive Summary")
            st.success(summary)

            c1, c2 = st.columns(2)

            with c1:
                kpi_card(
                    "Average Retrieved Risk Score",
                    round(avg_risk, 2)
                )

            with c2:
                kpi_card(
                    "Risk Level",
                    risk_level
                )

            st.subheader("Key Complaint Drivers")

            driver_df = (
                theme_counts
                .reset_index()
            )

            driver_df.columns = [
                "Theme",
                "Retrieved Complaints"
            ]

            st.dataframe(
                driver_df,
                use_container_width=True
            )

            st.subheader("Recommended Business Actions")

            for rec in recommendations[:5]:
                st.markdown(
                    f"- {rec}"
                )

            st.subheader("Supporting Complaint Evidence")

            for _, row in results.head(5).iterrows():
                with st.expander(
                    f"{row['theme']} | Risk Score: {round(row['risk_score'], 2)}"
                ):
                    st.write(
                        row["consumer_complaint_narrative"][:1500]
                    )

# -----------------------------
# Tab 4: High Risk Complaints
# -----------------------------
with tab4:
    st.markdown(
        '<div class="section-header">High-Risk Complaint Queue</div>',
        unsafe_allow_html=True
    )

    if len(filtered) == 0:
        st.warning("No data available for the selected filters.")
    else:
        risk_threshold = st.slider(
            "Minimum Risk Score",
            float(filtered["risk_score"].min()),
            float(filtered["risk_score"].max()),
            float(filtered["risk_score"].quantile(0.90))
        )

        high_risk = (
            filtered[
                filtered["risk_score"] >= risk_threshold
            ]
            .sort_values(
                "risk_score",
                ascending=False
            )
        )

        st.write(
            f"{len(high_risk):,} complaints above selected risk threshold."
        )

        cols_to_show = [
            "theme",
            "risk_score",
            "recommendation",
            "consumer_complaint_narrative"
        ]

        if "company" in filtered.columns:
            cols_to_show = [
                "company"
            ] + cols_to_show

        st.dataframe(
            high_risk[cols_to_show],
            use_container_width=True
        )
