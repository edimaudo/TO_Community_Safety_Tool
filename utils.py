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


def fetch_relevant_data(query_text, top_k=50):
    """Perform vector search in TiDB using precomputed embeddings"""
    conn = connect_tidb()
    if conn is None:
        return pd.DataFrame()

    try:
        # 1. Embed the user query
        resp = client.models.embed_content(
            model=EMBED_MODEL,
            contents=[query_text]
        )
        query_vec = resp.embeddings[0].values

        # 2. Run vector similarity search in TiDB
        sql = """
        SELECT id, occurrence_date, offence, mci_category, neighbourhood, location_type,
               ST_Distance_Sphere(embedding, CAST(%s AS VECTOR)) AS distance
        FROM crime_data
        ORDER BY distance ASC
        LIMIT %s
        """
        df = pd.read_sql(sql, conn, params=[json.dumps(query_vec), top_k])
        return df

    except Exception as e:
        st.error(f"Vector search failed: {e}")
        return pd.DataFrame()
    finally:
        conn.close()


def ask_gemini(question, context_df):
    """Ask Gemini using retrieved context"""
    try:
        schema = """
        Respond ONLY in JSON matching this schema:
        {
          "answer": "short explanation",
          "chart": {
            "type": "bar|line|pie|null",
            "x": "column name or null",
            "y": "column name or null",
            "color": "column name or null"
          }
        }
        """

        sample = context_df.to_dict(orient="records")

        prompt = f"""
        You are analyzing a crime dataset. 
        User question: {question}

        Relevant rows from database:
        {json.dumps(sample, indent=2)}

        Now provide an answer and suggest a visualization.
        {schema}
        """

        resp = client.models.generate_content(
            model=GEN_MODEL,
            contents=prompt
        )

        try:
            parsed = json.loads(resp.text)
            return parsed
        except Exception:
            return {"answer": resp.text, "chart": None}

    except Exception as e:
        return {"answer": f"Gemini API error: {e}", "chart": None}


def visualize_data(df, chart_spec):
    if not chart_spec or not isinstance(chart_spec, dict):
        return None

    chart_type = chart_spec.get("type")
    x_col = chart_spec.get("x")
    y_col = chart_spec.get("y")
    color_col = chart_spec.get("color")

    try:
        if chart_type == "bar":
            fig = px.bar(df, x=x_col, y=y_col, color=color_col, title="Crime Data")
        elif chart_type == "line":
            fig = px.line(df, x=x_col, y=y_col, color=color_col, title="Crime Data Trend")
        elif chart_type == "pie":
            fig = px.pie(df, names=x_col, values=y_col, title="Crime Data Distribution")
        else:
            return None
        return fig
    except Exception as e:
        st.error(f"Visualization error: {e}")
        return None