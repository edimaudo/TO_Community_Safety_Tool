from utils import *
from data import *

st.title(APP_NAME)
st.header(PREDICTON_HEADER)

st.write("Use the filters to predict that type of incidents and get insights into how to keep yourself safe")

with st.sidebar:
    month_options = st.selectbox('Month',MONTH)
    dow_options = st.selectbox('Day of Week',DAY_OF_WEEK)
    hour_options = st.selectbox('Hour',HOUR)
    premises_options = st.selectbox('Premises Type',PREMISES_TYPE)
    neighbourhood_options = st.selectbox('Neighborhood',NEIGHBORHOOD)
    clicked = st.button("Run Incident Prediction")

# Model setup
# Model information
model_data = df_filtered[['OCC_MONTH','OCC_DOW','OCC_HOUR','PREMISES_TYPE','Neighborhood','MCI_CATEGORY']]
# Categorize data
model_data["OCC_MONTH"] = model_data["OCC_MONTH"].astype('category')
model_data["OCC_DOW"] = model_data["OCC_DOW"].astype('category')
model_data["PREMISES_TYPE"] = model_data["PREMISES_TYPE"].astype('category')
model_data["Neighborhood"] = model_data["Neighborhood"].astype('category')
model_data["MCI_CATEGORY"] = model_data["MCI_CATEGORY"].astype('category')
model_data["OCC_MONTH_cat"] = model_data["OCC_MONTH"].cat.codes
model_data["OCC_DOW_cat"] = model_data["OCC_DOW"].cat.codes
model_data["PREMISES_TYPE_cat"] = model_data["PREMISES_TYPE"].cat.codes
model_data["Neighborhood_cat"] = model_data["Neighborhood"].cat.codes
model_data["MCI_CATEGORY_cat"] = model_data["MCI_CATEGORY"].cat.codes

model_info = model_data[(model_data.OCC_MONTH == month_options)]
model_info.reset_index(drop=True, inplace=True)
month = model_info['OCC_MONTH_cat'][0]

model_info = model_data[(model_data.OCC_DOW == dow_options)]
model_info.reset_index(drop=True, inplace=True)
dow = model_info['OCC_DOW_cat'][0]

model_info = model_data[(model_data.OCC_HOUR == hour_options)]
model_info.reset_index(drop=True, inplace=True)
hour = model_info['OCC_HOUR'][0]

model_info = model_data[(model_data.PREMISES_TYPE == premises_options)]
model_info.reset_index(drop=True, inplace=True)
premise = model_info['PREMISES_TYPE_cat'][0]

model_info = model_data[(model_data.Neighborhood == neighbourhood_options)]
model_info.reset_index(drop=True, inplace=True)
neighbourhood = model_info['Neighborhood_cat'][0]

if clicked:
    info_df = pd.DataFrame(columns = ['OCC_MONTH_cat','OCC_DOW_cat','OCC_HOUR','PREMISES_TYPE_cat','Neighborhood_cat'],index = ['a'])
    info_df.loc['a'] = [month,dow,hour, premise, neighbourhood]
    # Load model
    saved_final_model = load_model('model/Final Model')
    # Prediction
    new_prediction = predict_model(saved_final_model, data=info_df)
    crime_category = new_prediction['prediction_label'][0]
    if crime_category == 0:
        crime_output = 'Assault'
    elif crime_category == 1:
        crime_output = 'Auto Theft'
    elif crime_category == 2:
        crime_output = 'Break & Enter'
    elif crime_category == 3:
        crime_output = 'Robbery'
    else:
        crime_output = 'Theft Over'
    st.write(" ")
    st.write("For the chosen selection the Predicted Incident Category is: " + crime_output)
    
    # Generate Recommendations using LLMs
    st.subheader("Neighhourhood Action Steps")
    st.write("Based on the incident category: " + crime_output + ', here are some safety recommendations for the neighborhood')
    prompt = "Generate the output using a numbered bullet point format.  You are a neighourhood safety advisor. Don't show the neighbourhood safety advisor information. Based on the following crime" + str(crime_output) + " that occurred in " + str(premises_options) + " at " + str(hour_options) + " hours in " + str(neighbourhood_options) + " a neigbhorhood in Toronto, Ontario, " + "generate 3 personalized practical safety recommendations for local residents."
    
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


