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
    neighbourhood=None,
    years=None,
    months=None,
    dows=None,
    categories=None,
    premises=None,
):
    """
    Hybrid vector + structured search.

    Accepts either single values or lists for filters. Builds safe SQL clauses:
      - if a filter has one value -> uses "col = %s"
      - if a filter has multiple values -> uses "col IN (%s,%s,...)"

    Returns: pandas.DataFrame
    """
    try:
        # 1) get embedding
        query_emb = get_embedding(query)
        emb_json = json.dumps(query_emb)

        # 2) normalize inputs -> ensure lists or None
        def _norm(v):
            if v is None:
                return None
            # If it's a scalar (string/int), wrap in list
            if isinstance(v, (str, int)):
                return [v]
            # If it's a pandas Series/Index, convert to list and dropna
            try:
                import pandas as _pd
                if isinstance(v, (_pd.Series, _pd.Index)):
                    return [x for x in list(v) if pd.notna(x)]
            except Exception:
                pass
            # If it's iterable (like list/tuple), convert to list and filter NA
            if hasattr(v, "__iter__"):
                return [x for x in list(v) if x is not None and (not (isinstance(x, float) and pd.isna(x)))]
            return [v]

        neighbourhood = _norm(neighbourhood)
        years = _norm(years)
        months = _norm(months)
        dows = _norm(dows)
        categories = _norm(categories)
        premises = _norm(premises)

        # 3) build WHERE clauses safely
        conditions = ["embedding IS NOT NULL"]
        params = [emb_json]

        def _add_filter(col_name, values):
            # values is a list (or None)
            if not values:
                return
            # single value
            if len(values) == 1:
                conditions.append(f"{col_name} = %s")
                params.append(values[0])
            else:
                placeholders = ", ".join(["%s"] * len(values))
                conditions.append(f"{col_name} IN ({placeholders})")
                params.extend(values)

        _add_filter("occ_year", years)
        _add_filter("occ_month", months)
        _add_filter("occ_dow", dows)
        _add_filter("mci_category", categories)
        _add_filter("premises_type", premises)
        _add_filter("neighborhood", neighbourhood)

        where_clause = " AND ".join(conditions)

        sql = f"""
        SELECT
          id, occ_date, occ_year, occ_month, occ_day, occ_dow, occ_hour,
          mci_category, offence, neighborhood, premises_type,
          VEC_COSINE_DISTANCE(embedding, CAST(%s AS VECTOR)) AS score
        FROM crime_data
        WHERE {where_clause}
        ORDER BY score ASC
        LIMIT %s;
        """

        # append top_k as last param
        params.append(top_k)

        conn = connect_tidb()
        cursor = conn.cursor(pymysql.cursors.DictCursor)
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
