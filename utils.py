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
import datetime
import time
import numpy as np
import statistics
import xgboost
import pmdarima as pm
from google import genai
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


