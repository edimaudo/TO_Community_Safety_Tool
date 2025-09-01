from utils import *

pg = st.navigation([
    st.Page("About.py"),
    st.Page("Overview.py"),
    st.Page("Neighbourhood_Incident_Exploration.py"),
    st.Page("Neighbourhood_Incident_Predictor.py"),
    st.Page("Neighourhood_Incident_Forecast.py"),
])
pg.run()