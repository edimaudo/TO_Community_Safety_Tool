from utils import * 
from data import *

st.title(APP_NAME)
st.header(NEIGHBORHOOD_HEADER)

st.write("""
         Highlights Socio-economic Metrics and Incident trends for the selected neighborhood in Toronto, Ontario.  
         """)


tab1, tab2= st.tabs(["Neighourhood Socio-Ecomonic Metrics", "Neighourhood Incident Trends"])
with tab1:
    with st.sidebar:
        neighourhood_options = st.selectbox('Neighbourhood',NEIGHBORHOOD)
        year_options = st.multiselect('Year',YEAR,default=YEAR)
        month_options = st.multiselect('Month',MONTH,default=MONTH)
        dow_options = st.multiselect('Day of Week',DAY_OF_WEEK,default=DAY_OF_WEEK)
        mci_options = st.multiselect('Crime Type',MCI_CATEGORY,default=MCI_CATEGORY)
        premises_options = st.multiselect('Premises Type',PREMISES_TYPE,default=PREMISES_TYPE) 
    
    st.subheader("Culture")
    col1, col2 = st.columns(2)
    wellbeing_culture = wellbeing_culture[(wellbeing_culture['Neighbourhood'] == neighourhood_options)]
    col1.metric("Lingustic Diversity", round(statistics.median(wellbeing_culture['Linguistic Diversity Index']),2))
    col2.metric("Cultural Location Index", statistics.median(wellbeing_culture['Cultural Location Index']))

    st.subheader("Economics")
    col1, col2, col3, col4, col5 = st.columns(5)
    wellbeing_economics = wellbeing_economics[(wellbeing_economics['Neighbourhood'] == neighourhood_options)]
    col1.metric("No. of Businessess",statistics.median(wellbeing_economics['Businesses']))
    col2.metric("Child Care Spaces",statistics.median(wellbeing_economics['Child Care Spaces']))
    col3.metric("Debt Risk Score",statistics.median(wellbeing_economics['Debt Risk Score']))
    col4.metric("Local Employment",statistics.median(wellbeing_economics['Local Employment']))
    col5.metric("Social Assistance Recipients",statistics.median(wellbeing_economics['Social Assistance Recipients']))

    st.subheader("Environment")
    col1, col2 = st.columns(2)
    wellbeing_environment = wellbeing_environment[(wellbeing_environment['Neighbourhood'] == neighourhood_options)]
    col1.metric("No. of Green Rebate Programs",statistics.median(wellbeing_environment['Green Rebate Programs']))
    col2.metric("Green Spaces (Square Kilometres)", round(statistics.median(wellbeing_environment['Green Spaces']),2))

    st.subheader("Health")
    col1, col2 = st.columns(2)
    wellbeing_health = wellbeing_health[(wellbeing_health['Neighbourhood'] == neighourhood_options)]
    col1.metric("No. of DineSafe Inspections",statistics.median(wellbeing_health["DineSafe Inspections"]))
    col2.metric("# of Health Providers",statistics.median(wellbeing_health['Health Providers']))

with tab2:

    with st.container():
        col1, col2, col3 = st.columns(3)
        neighorhood_df = df_filtered[(df_filtered['OCC_YEAR'].isin(year_options)) & (df_filtered['OCC_MONTH'].isin(month_options)) & (df_filtered['OCC_DOW'].isin(dow_options)) & (df_filtered['MCI_CATEGORY'].isin(mci_options)) & (df_filtered['PREMISES_TYPE'].isin(premises_options)) & (df_filtered['Neighborhood'] == neighourhood_options)]
        with col1:
            st.subheader("Total Crime by Year")
            crimes_data = neighorhood_df[['MCI_CATEGORY','OCC_YEAR']]
            crimes_data = crimes_data.groupby(['OCC_YEAR']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
            crimes_data.columns = ['Year','Total']
            crimes_data.sort_values("Year", ascending=True)
            fig = px.bar(crimes_data, x="Year", y="Total")
            st.plotly_chart(fig)
        with col2:
            st.subheader("Crime Category by Year")
            crimes_data = neighorhood_df[['MCI_CATEGORY','OCC_YEAR']]
            crimes_data = crimes_data.groupby(['MCI_CATEGORY','OCC_YEAR']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
            crimes_data.columns = ['MCI CATEGORY', 'Year','Total']
            crimes_data.sort_values("Year", ascending=True)
            fig = px.line(crimes_data, x="Year", y="Total",color='MCI CATEGORY')
            st.plotly_chart(fig)
        with col3:
            st.subheader("MCI Category Mix")
            crimes_data = neighorhood_df[['MCI_CATEGORY']]
            crimes_data = crimes_data.groupby(['MCI_CATEGORY']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
            crimes_data.columns = ['MCI CATEGORY','Total']   

            fig = px.pie(values = crimes_data['Total'],
                        names = crimes_data['MCI CATEGORY'],
                        color = crimes_data['MCI CATEGORY'],
                        hole = 0.5)
            st.plotly_chart(fig)


    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Crime Category by Month")
        crimes_data = neighorhood_df[['MCI_CATEGORY','OCC_MONTH']]
        crimes_data = crimes_data.groupby(['MCI_CATEGORY','OCC_MONTH']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
        crimes_data.columns = ['MCI CATEGORY', 'Month','Total']
        month_dict = {'January':1,'February':2,'March':3, 'April':4, 'May':5, 'June':6, 'July':7, 
        'August':8, 'September':9, 'October':10, 'November':11, 'December':12}
        crimes_data = crimes_data.sort_values('Month', key = lambda x : x.apply (lambda x : month_dict[x]))
        fig = px.line(crimes_data, x="Month", y="Total",color='MCI CATEGORY')
        st.plotly_chart(fig)

        st.subheader("Crime Category by Day of Week")
        crimes_data = neighorhood_df[['MCI_CATEGORY','OCC_DOW']]
        dow_dict = {'Monday':1,'Tuesday':2,'Wednesday':3, 'Thursday':4, 'Friday':5, 'Saturday':6, 'Sunday':7}
        crimes_data = crimes_data[crimes_data['OCC_DOW'].isin(dow_dict)]
        crimes_data = crimes_data.groupby(['MCI_CATEGORY','OCC_DOW']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
        crimes_data.columns = ['MCI CATEGORY', 'Day of Week','Total']
        crimes_data = crimes_data.sort_values('Day of Week', key = lambda x : x.apply (lambda x : dow_dict[x]))
        fig = px.line(crimes_data, x="Day of Week", y="Total",color='MCI CATEGORY')
        st.plotly_chart(fig)

        st.subheader("Crime by Hour & Day of Week")

        crimes_data = neighorhood_df[['OCC_HOUR', 'OCC_DOW']]
        crimes_data = (
            crimes_data
            .groupby(['OCC_HOUR', 'OCC_DOW'])
            .agg(Total_reviews=('OCC_DOW', 'count'))
            .reset_index()
        )
        crimes_data.columns = ['Hour', 'Day of Week', 'Total']

        dow_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

        matrix = (
            crimes_data
            .pivot(index='Day of Week', columns='Hour', values='Total')
            .reindex(dow_order)  # re-order index
            .fillna(0)
        )

        fig = px.imshow(
            matrix.values,
            x=matrix.columns,
            y=matrix.index,
            color_continuous_scale='Viridis'
        )

        fig.update_layout(width=500, height=500)
        st.plotly_chart(fig)



    with col2:
        st.subheader("Crime Category by Day of the Month")
        crimes_data = neighorhood_df[['MCI_CATEGORY','OCC_DAY']]
        crimes_data = crimes_data.groupby(['MCI_CATEGORY','OCC_DAY']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
        crimes_data.columns = ['MCI CATEGORY', 'Day','Total']
        crimes_data.sort_values("Day", ascending=True)
        fig = px.line(crimes_data, x="Day", y="Total",color='MCI CATEGORY')
        st.plotly_chart(fig)

        st.subheader("Crime Category by Hour")
        crimes_data = neighorhood_df[['MCI_CATEGORY','OCC_HOUR']]
        crimes_data = crimes_data.groupby(['MCI_CATEGORY','OCC_HOUR']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
        crimes_data.columns = ['MCI CATEGORY', 'Hour','Total']
        crimes_data.sort_values("Hour", ascending=True)
        fig = px.line(crimes_data, x="Hour", y="Total",color='MCI CATEGORY')
        st.plotly_chart(fig)

        st.subheader("Crime Category by Premise Type")
        crimes_data = neighorhood_df[['MCI_CATEGORY','PREMISES_TYPE']]
        crimes_data = crimes_data.groupby(['MCI_CATEGORY','PREMISES_TYPE']).agg(Total_reviews = ('MCI_CATEGORY', 'count')).reset_index()
        crimes_data.columns = ['MCI CATEGORY', 'PREMISES TYPE','Total']
        # Pivot so MCI_CATEGORY is columns, PREMISES_TYPE is rows
        matrix = crimes_data.pivot(index='PREMISES TYPE', columns='MCI CATEGORY', values='Total').fillna(0)

        fig = px.imshow(
            matrix.values,                      # the numeric matrix
            x=matrix.columns,                    # categories on x-axis
            y=matrix.index,                      # premises on y-axis
            color_continuous_scale='Viridis'
        )

        fig.update_layout(width=500, height=500)
        st.plotly_chart(fig)
 


