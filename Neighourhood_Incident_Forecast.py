from utils import *
from data import *

st.title(APP_NAME)
st.header(FORECAST_HEADER)

st.write("Use the filters to forecast future incidents and get insights into potential future incidents and how to keep yourself safe")

with st.sidebar:
        neighourhood_options = st.selectbox('Neighbourhood',NEIGHBORHOOD)
        mci_options = st.selectbox('Crime Type',MCI_CATEGORY)
        forecast_horizon = st.slider("Forecast Range (months)", 12, 36, 12, step=6)
        clicked = st.button("Generate Incident Forecast")

# Setup data
neighorhood_df = df_filtered[(df_filtered['MCI_CATEGORY'] == (mci_options)) & (df_filtered['Neighborhood'] == neighourhood_options)]

# Forecast setup
# Aggregate to monthly counts
model_data = neighorhood_df[['OCC_YEAR','OCC_MONTH','Neighborhood','MCI_CATEGORY']]
crime_monthly = (
    model_data
    .groupby(['OCC_YEAR','OCC_MONTH','Neighborhood','MCI_CATEGORY'])
    .size()
    .reset_index(name="Total")
)

# Build a proper monthly Date column and sort
crime_monthly['DATE'] = pd.to_datetime(
    crime_monthly['OCC_YEAR'].astype(str) + "-" +
    crime_monthly['OCC_MONTH'].astype(str).str.zfill(2) + "-01"
)
crime_monthly = crime_monthly.sort_values('DATE')


def forecast_category(crime_monthly: pd.DataFrame, forecast_horizon: int) -> pd.DataFrame:
    """
    Fit Auto-ARIMA on monthly totals and return a DataFrame with future dates and forecasts.
    Expects crime_monthly with columns: DATE (datetime), Total (numeric).
    """

    if crime_monthly is None or crime_monthly.empty:
        return pd.DataFrame(columns=['DATE', 'Forecast'])

    # Build continuous monthly series (fill missing months with 0 counts)
    ts = (
        crime_monthly
        .set_index('DATE')['Total']
        .asfreq('MS')             # monthly frequency
        .fillna(0.0)              # fill gaps as zero counts
        .astype(float)
        .sort_index()
    )

    # Fit auto_arima model
    model = pm.auto_arima(
        ts,
        seasonal=True,
        m=12,                     # seasonality = 12 months
        stepwise=True,
        suppress_warnings=True,
        error_action='ignore',
        trace=False
    )

    # Forecast horizon months
    preds = model.predict(n_periods=forecast_horizon)

    # Future date index
    future_index = pd.date_range(
        ts.index[-1] + pd.offsets.MonthBegin(1),
        periods=forecast_horizon,
        freq="MS"
    )

    forecast_df = pd.DataFrame({
        'DATE': future_index,
        'Forecast': np.maximum(preds, 0.0)  # ensure no negatives
    })

    return forecast_df


# Run forecast
if clicked:
     forecast_df = forecast_category(crime_monthly, forecast_horizon)
     fig = px.bar(
        forecast_df,
        x="DATE",
        y="Forecast",
        title=f"{mci_options} incidents — {forecast_horizon}-Month Forecast for {neighourhood_options}",
        labels={"DATE": "Date", "Forecast": "Forecasted Incidents"}
    )
     fig.update_layout(template="plotly_white", xaxis_title="Date", yaxis_title="Total")
     st.plotly_chart(fig, use_container_width=True)

     # Generate recommendations using Gemini
     st.subheader("Neighhourhood Action Steps")
     st.write("Based on the incident category: " + mci_options + ', here are some safety recommendations for ' + str(neighourhood_options))
     prompt = "Generate the output using a numbered bullet point format.  You are a neighourhood safety advisor. Don't show the neighbourhood safety advisor information.  Based on the following crime " + "This is the " + str(forecast_horizon) + " forecast with the data " + str(forecast_df['Forecast'].to_list()) + " for " + mci_options + " that occurred in " + str(neighourhood_options) + " a neigbhorhood in Toronto, Ontario, " + "generate 3 personalized practical safety recommendations for local residents."

     @st.cache_resource
     def prompt_generation(prompt_text):
        client = genai.Client() 
        response = client.models.generate_content(
        model="gemini-2.5-flash", 
        contents=prompt_text
        )
        return response
    
     prompt_output = prompt_generation(prompt)

     outcome_txt = st.text_area(label=" ",value=prompt_output.text,placeholder='', disabled=True)



