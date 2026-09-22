from __future__ import annotations

import json
import os
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from sqlalchemy import create_engine

st.set_page_config(page_title="Support Search Monitoring", layout="wide")
st.title("Support Search Monitoring")
database = Path(os.getenv("SEARCH_DATABASE_PATH", "artifacts/search.db"))
database_url = os.getenv("SEARCH_DATABASE_URL")
if not database_url and not database.exists():
    st.info("No traffic recorded yet. Query the API to populate this dashboard.")
    st.stop()
engine = create_engine(database_url or f"sqlite:///{database.resolve().as_posix()}")
with engine.connect() as db:
    queries = pd.read_sql_query("SELECT * FROM queries ORDER BY created_at", db)
    feedback = pd.read_sql_query("SELECT * FROM feedback ORDER BY created_at", db)
if queries.empty:
    st.info("The event database is initialized but contains no queries.")
    st.stop()
queries["event_metadata"] = queries.event_metadata.map(json.loads)
queries["reranked"] = queries.event_metadata.map(lambda row: row.get("reranked", False))
c1, c2, c3, c4 = st.columns(4)
c1.metric("Queries", len(queries))
c2.metric("P95 latency", f"{queries.latency_ms.quantile(0.95):.1f} ms")
c3.metric("Failed query rate", f"{queries.failed.mean():.1%}")
c4.metric("Feedback coverage", f"{len(feedback) / len(queries):.1%}")
st.plotly_chart(
    px.histogram(queries, x="latency_ms", color="mode", title="Latency distribution"),
    use_container_width=True,
)
hourly = (
    queries.assign(time=pd.to_datetime(queries.created_at, unit="s", utc=True))
    .set_index("time")
    .resample("1h")
    .agg(latency_ms=("latency_ms", "mean"), failed_rate=("failed", "mean"))
)
st.plotly_chart(
    px.line(hourly, y=["latency_ms", "failed_rate"], title="Operational drift"),
    use_container_width=True,
)
if not feedback.empty:
    st.plotly_chart(
        px.histogram(feedback, x="relevance", title="Human relevance feedback"),
        use_container_width=True,
    )
st.subheader("Recent failed query hashes")
st.dataframe(queries[queries.failed == 1].tail(25), use_container_width=True)
