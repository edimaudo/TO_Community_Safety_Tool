"""
Libraries
"""
import streamlit as st
import pandas as pd
import numpy as np
import datetime
import os, os.path
import warnings
import random
import plotly.express as px
import plotly.graph_objects as go
import geopandas as gpd
import folium
import pickle
from pycaret.classification import *
import pymysql
import json
import datetime
import time
import numpy as np
import statistics
import xgboost
import pmdarima as pm
from google import genai
import ssl
from dotenv import load_dotenv, dotenv_values 
load_dotenv() 

"""
App Information
"""
APP_NAME = 'TO Neighbourhood Safety Tool'
ABOUT_HEADER = 'About'
OVERVIEW_HEADER = 'Socio-economic Metrics & Incidents Overview'
NEIGHBORHOOD_HEADER = "Neighborhood Socio-economic Metrics & Incidents"
PREDICTON_HEADER = 'Neighborhood Incidents Prediction'
FORECAST_HEADER = 'Neighborhood Incidents Forecast'
APP_FILTERS = 'Filters'
NO_DATA_INFO = 'No data available to display based on the filters'

warnings.simplefilter(action='ignore', category=FutureWarning)
st.set_page_config(
    page_title=APP_NAME,
    layout="wide"
)

# ----------------- TIDB setup -----------------
TIDB_HOST = os.getenv("TIDB_HOST")
TIDB_PORT = int(os.getenv("TIDB_PORT", 4000))
TIDB_USER = os.getenv("TIDB_USER")
TIDB_PASSWORD = os.getenv("TIDB_PASSWORD")
TIDB_DATABASE = os.getenv("TIDB_DATABASE")
TIDB_SSL_CA = os.getenv("TIDB_SSL_CA")

SSL_VERIFY_CERT = os.getenv("ssl_verify_cert", "True").lower() == "true"
SSL_VERIFY_IDENTITY = os.getenv("ssl_verify_identity", "True").lower() == "true"

ssl_config = {}
if TIDB_SSL_CA and os.path.exists(TIDB_SSL_CA):
    cert_reqs = ssl.CERT_REQUIRED if SSL_VERIFY_CERT else ssl.CERT_NONE
    ssl_config = {"ca": TIDB_SSL_CA, "cert_reqs": cert_reqs}
else:
    ssl_config = {"ssl": True}

# ----------------- Gemini Setup -----------------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GEN_MODEL = "models/gemini-2.5-flash"
EMBED_MODEL = "models/text-embedding-004"
client = genai.Client(api_key=GEMINI_API_KEY)

# ----------------- Helpers -----------------
def connect_tidb():
    try:
        conn = pymysql.connect(
            host=TIDB_HOST,
            port=TIDB_PORT,
            user=TIDB_USER,
            password=TIDB_PASSWORD,
            database=TIDB_DATABASE,
            ssl=ssl_config,
            connect_timeout=20
        )
        return conn
    except Exception as e:
        st.error(f"Database connection failed: {e}")
        return None


# ----------------- Embedding helper -----------------
def get_embedding(text: str):
    """Get embedding vector from Gemini."""
    if not text.strip():
        return [0.0] * 768
    resp = client.models.embed_content(model=EMBED_MODEL, contents=text)
    return list(resp.embeddings[0].values)

# ----------------- Vector search -----------------
def vector_search(
    query: str,
    top_k: int = 20,
    neighbourhood: list = None,
    years: list = None,
    months: list = None,
    dows: list = None,
    categories: list = None,
    premises: list = None,
):
    """
    Hybrid vector + structured search.
    - query: free text for embedding similarity
    - top_k: number of rows to return
    - optional filters: if lists provided, filter SQL results
    """
    try:
        query_emb = get_embedding(query)
        emb_json = json.dumps(query_emb)

        conn = connect_tidb()
        cursor = conn.cursor(pymysql.cursors.DictCursor)

        conditions = ["embedding IS NOT NULL"]
        params = [emb_json]

        if years and len(years) > 0:
            conditions.append("occ_year IN %s")
            params.append(tuple(years))

        if months and len(months) > 0:
            conditions.append("occ_month IN %s")
            params.append(tuple(months))

        if dows and len(dows) > 0:
            conditions.append("occ_dow IN %s")
            params.append(tuple(dows))

        if categories and len(categories) > 0:
            conditions.append("mci_category IN %s")
            params.append(tuple(categories))

        if premises and len(premises) > 0:
            conditions.append("premises_type IN %s")
            params.append(tuple(premises))

        if neighbourhood and len(neighbourhood) > 0:
            conditions.append("neighborhood IN %s")
            params.append(tuple(neighbourhood))

        where_clause = " AND ".join(conditions)

        sql = f"""
        SELECT id, occ_date, occ_year, occ_month, occ_day, occ_dow, occ_hour,
               mci_category, offence, neighborhood, premises_type,
               VEC_COSINE_DISTANCE(embedding, CAST(%s AS VECTOR)) AS score
        FROM crime_data
        WHERE {where_clause}
        ORDER BY score ASC
        LIMIT {top_k};
        """

        cursor.execute(sql, tuple(params))
        rows = cursor.fetchall()
        cursor.close()
        conn.close()

        return pd.DataFrame(rows)

    except Exception as e:
        st.error(f"Vector search error: {e}")
        return pd.DataFrame()


# ----------------- Ask Gemini -----------------
def ask_gemini(query: str, df: pd.DataFrame, filters: dict = None):
    """
    Use Gemini to answer based on vector search results.
    filters: optional dict to describe applied filters (year, neighbourhood, etc.)
    """
    try:
        if df.empty:
            return "No matching incidents found in the dataset."

        # Build context string from the dataframe
        context = df.to_dict(orient="records")

        # Build a filter summary for the model
        filter_text = ""
        if filters:
            applied = [f"{k}: {', '.join(map(str, v))}" for k, v in filters.items() if v]
            if applied:
                filter_text = "\nFilters applied: " + "; ".join(applied)

        prompt = f"""
        You are analyzing Toronto crime incident data. 
        Here are the most relevant records (JSON): {json.dumps(context, default=str)}
        {filter_text}

        Question: {query}

        Based only on this data:
        - Provide a clear, text-based answer (not JSON).
        - Summarize patterns, counts, or trends where possible.
        - Mention if filters were applied so the user understands context.
        """

        resp = client.models.generate_content(
            model="models/gemini-2.5-flash",
            contents=prompt,
        )

        return resp.text.strip()

    except Exception as e:
        return f"Gemini API error: {e}"
