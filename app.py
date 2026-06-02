#!/usr/bin/env python
# coding: utf-8

# In[ ]:
 #!/usr/bin/env python
# coding: utf-8

import streamlit as st
import pandas as pd
import plotly.express as px
from collections import Counter

st.set_page_config(
    page_title="AI Complaint Intelligence Platform",
    layout="wide"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"]  {
    font-family: 'Poppins', sans-serif;
}
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("processed_complaints.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["risk_score"] = pd.to_numeric(df["risk_score"], errors="coerce")
    df["consumer_complaint_narrative"] = df["consumer_complaint_narrative"].astype(str)

    required_cols = ["theme", "risk_score", "recommendation", "consumer_complaint_narrative"]
    return df.dropna(subset=required_cols)

pdf = load_data()

st.title("AI-Powered Customer Complaint Intelligence Platform")
st.caption("NLP + Risk Scoring + Semantic Retrieval + Executive Recommendations")

# -----------------------------
# Helper Functions
# -----------------------------
def retrieve_relevant_complaints(query, data, k=10):
    query_words = query.lower().split()

    temp = data.copy()
    temp["match_score"] = temp["consumer_complaint_narrative"].astype(str).apply(
        lambda x: sum(word in x.lower() for word in query_words)
    )

    results = (
        temp.sort_values(["match_score", "risk_score"], ascending=False)
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
        f"{', '.join(top_themes)}. The average risk score is {avg_risk:.2f}, "
        f"indicating a {risk_level.lower()} level of business risk."
    )

    return results, summary, theme_counts, avg_risk, risk_level, recommendations


# -----------------------------
# Sidebar Filters
# -----------------------------
st.sidebar.header("Filters")

if "company" in pdf.columns:
    company_options = ["All"] + sorted(pdf["company"].dropna().astype(str).unique().tolist())
else:
    company_options = ["All"]

company_filter = st.sidebar.selectbox("Company", company_options)

theme_filter = st.sidebar.selectbox(
    "Theme",
    ["All"] + sorted(pdf["theme"].dropna().unique().tolist())
)

filtered = pdf.copy()

if company_filter != "All" and "company" in filtered.columns:
    filtered = filtered[filtered["company"].astype(str) == company_filter]

if theme_filter != "All":
    filtered = filtered[filtered["theme"] == theme_filter]

# -----------------------------
# KPI Cards
# -----------------------------
col1, col2, col3, col4 = st.columns(4)

col1.metric("Complaints Analyzed", f"{len(filtered):,}")
col2.metric("Themes Discovered", filtered["theme"].nunique())
col3.metric("Avg Risk Score", round(filtered["risk_score"].mean(), 2))

highest_risk_theme = (
    filtered.groupby("theme")["risk_score"].mean().idxmax()
    if len(filtered) > 0 else "N/A"
)

col4.metric("Highest Risk Theme", highest_risk_theme)

st.divider()

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
    st.subheader("Executive Overview")

    left, right = st.columns(2)

    theme_volume = filtered["theme"].value_counts().reset_index()
    theme_volume.columns = ["Theme", "Complaint Count"]

    fig1 = px.bar(
        theme_volume,
        x="Complaint Count",
        y="Theme",
        orientation="h",
        title="Complaint Volume by Theme"
    )
    fig1.update_layout(yaxis={"categoryorder": "total ascending"})
    left.plotly_chart(fig1, use_container_width=True)

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
    fig2.update_layout(yaxis={"categoryorder": "total ascending"})
    right.plotly_chart(fig2, use_container_width=True)

    st.subheader("Recommendation Summary")

    rec_summary = (
        filtered.groupby(["theme", "recommendation"])
        .size()
        .reset_index(name="complaint_count")
        .sort_values("complaint_count", ascending=False)
    )

    st.dataframe(rec_summary, use_container_width=True)

# -----------------------------
# Tab 2: Theme Analysis
# -----------------------------
with tab2:
    st.subheader("Explore Complaint Themes")

    selected_theme = st.selectbox(
        "Select a theme to explore",
        sorted(filtered["theme"].dropna().unique())
    )

    theme_df = filtered[filtered["theme"] == selected_theme]

    c1, c2, c3 = st.columns(3)
    c1.metric("Complaints", f"{len(theme_df):,}")
    c2.metric("Avg Risk", round(theme_df["risk_score"].mean(), 2))
    c3.metric("Max Risk", round(theme_df["risk_score"].max(), 2))

    st.markdown("### Recommended Action")
    st.info(theme_df["recommendation"].mode().iloc[0])

    st.markdown("### Sample Complaints")
    for _, row in theme_df.head(5).iterrows():
        with st.expander(f"Risk Score: {round(row['risk_score'], 2)}"):
            st.write(row["consumer_complaint_narrative"][:1200])

# -----------------------------
# Tab 3: AI Agent
# -----------------------------
with tab3:
    st.header("🤖 AI Complaint Intelligence Agent")

    st.markdown(
        """
        Ask a business question and the system will retrieve relevant complaints, 
        summarize key drivers, assess risk, and recommend business actions.
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

    if st.button("Generate Executive Summary"):

        results, summary, theme_counts, avg_risk, risk_level, recommendations = answer_business_question(
            question,
            filtered,
            k
        )

        st.subheader("Executive Summary")
        st.success(summary)

        col1, col2 = st.columns(2)
        col1.metric("Average Retrieved Risk Score", round(avg_risk, 2))
        col2.metric("Risk Level", risk_level)

        st.subheader("Key Complaint Drivers")

        driver_df = theme_counts.reset_index()
        driver_df.columns = ["Theme", "Retrieved Complaints"]

        st.dataframe(driver_df, use_container_width=True)

        st.subheader("Recommended Business Actions")

        for rec in recommendations[:5]:
            st.markdown(f"- {rec}")

        st.subheader("Supporting Complaint Evidence")

        for _, row in results.head(5).iterrows():
            with st.expander(
                f"{row['theme']} | Risk Score: {round(row['risk_score'], 2)}"
            ):
                st.write(row["consumer_complaint_narrative"][:1500])

# -----------------------------
# Tab 4: High Risk Complaints
# -----------------------------
with tab4:
    st.subheader("High-Risk Complaint Queue")

    risk_threshold = st.slider(
        "Minimum Risk Score",
        float(filtered["risk_score"].min()),
        float(filtered["risk_score"].max()),
        float(filtered["risk_score"].quantile(0.90))
    )

    high_risk = (
        filtered[filtered["risk_score"] >= risk_threshold]
        .sort_values("risk_score", ascending=False)
    )

    st.write(f"{len(high_risk):,} complaints above selected risk threshold.")

    cols_to_show = [
        "theme",
        "risk_score",
        "recommendation",
        "consumer_complaint_narrative"
    ]

    if "company" in filtered.columns:
        cols_to_show = ["company"] + cols_to_show

    st.dataframe(
        high_risk[cols_to_show],
        use_container_width=True
    )
