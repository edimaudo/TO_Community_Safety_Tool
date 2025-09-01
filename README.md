# TO Neighborhood Safety Tool


   Focusing on Community Safety, this project analyzes crime and socio-economic data to understand incidents for Toronto, Ontario neighbourhoods.  
    It also enables the user to predict and forecast incidents for particular neighourhood.  
    Based on the predictions and forecast it would leverage Gemini AI to generate safety tips. 
    It is built using [streamlit](https://streamlit.io/cloud) and [Gemini AI](https://ai.google.dev/gemini-api/docs). The data is from [Toronto Police](https://data.torontopolice.on.ca/datasets/TorontoPS::major-crime-indicators-open-data/about) and [City of Toronto](https://data.urbandatacentre.ca/organization/city-of-toronto-open-data?q=wellbeing&sort=score+desc%2C+metadata_modified+desc&page=1).


    
### How to run it on your own machine

1. Install the requirements

   ```
   $ pip install -r requirements.txt
   ```

2. Run the app

   ```
   $ streamlit run streamlit_app.py
   ```
