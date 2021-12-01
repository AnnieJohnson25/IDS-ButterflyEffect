import streamlit as st
import pandas as pd
import plotly.express as px
from PIL import Image
from datetime import datetime

st.title('COVID-19 Out of the box viewpoints')

image = Image.open('out-of-the-box.jpg')
st.image(image, caption='COVID-19 Out of the box viewpoints')
 
########################################## LOADING DATA #################################################

@st.cache
def load_data():
    month_dict = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct",
                  11:"Nov", 12:"Dec"}
    df_brief = pd.read_csv("FlightsDepartureCount.csv")
    df_brief['Date'] = pd.to_datetime(df_brief["Date"])
    df_brief['Year'] =  pd.DatetimeIndex(df_brief["Date"]).year
    df_brief['Month #'] = pd.DatetimeIndex(df_brief["Date"]).month
    df_brief['Month'] = df_brief["Month #"]
    df_brief = df_brief.replace({"Month": month_dict})

    df_detailed = pd.read_csv("FlightsDepartureWithCountry.csv")
    df_detailed['Date'] = pd.to_datetime(df_detailed["Date"])
    df_detailed['Year'] = pd.DatetimeIndex(df_detailed["Date"]).year
    df_detailed['Month #'] = pd.DatetimeIndex(df_detailed["Date"]).month
    df_detailed['Month'] = df_brief["Month #"]
    df_detailed = df_detailed.replace({"Month": month_dict})

    return df_brief, df_detailed

df_brief, df_detailed = load_data()


st.sidebar.markdown("# [About] (#about)")
st.sidebar.markdown("# [Visualizations] (#visualizations)")
st.sidebar.markdown("# [Dataset](#dataset)")
st.sidebar.markdown("# [Advanced Machine Learning](#advanced-machine-learning)")
st.sidebar.markdown("# [Exploratory Data Analysis](#exploratory-data-analysis)")

########################################## ABOUT #################################################

st.title('About')

st.header('Problem Statement')

st.write("This project focuses on comparing some overlooked parameters and metrics that changed for good during the pandemic. This project specifically focuses on \n "
         "1. Gaming trend \n 2. Flight travel and carbon emissions \n 3. Suicide rate in the United States of America \n "
         "4. Climate change across the globe")

st.header('Demo')
st.video('https://youtu.be/fDek6cYijxI')

st.header('Project Team')

st.title('Visualizations')

########################################## GAMING DATA #################################################

st.header('Gaming Data')

image_gaming = Image.open('gaming.jpg')
st.image(image_gaming, width = 400)


########################################## SUICIDE DATA #################################################

st.header('Suicide Data')

image_suicide = Image.open('suicide.jpeg')
st.image(image_suicide, width = 400)

########################################## FLIGHTS DATA #################################################

st.header('Flights Data')

image_flights = Image.open('flights.jpg')
st.image(image_flights, width = 400)

#########################   Visualization Number 1 ########################

st.subheader('US Outgoing flight counts trend for years 1990 - 2020')
#over the years, departure count
#sheet = df_brief, columns = ['Total', 'Date']

df1 = pd.DataFrame(df_brief.groupby(by=['Year'])['Total Departures Count'].sum())
df1.reset_index(inplace=True)
df1['Total Departures Count'] = df1['Total Departures Count']/12
df1.loc[df1.Year == 2020,'Total Departures Count'] = df1.loc[df1.Year == 2020,'Total Departures Count'].item() * 12 / 3
ax = px.line(df1, x="Year", y="Total Departures Count", title='Outgoing Flights Count from US from 1990 to 2020')
st.plotly_chart(ax)

#########################   Visualization Number 2 ########################
st.subheader('Outgoing flights monthly traffic over the last 30 years up-until COVID')
#over the years, departure count, monthly trend
#sheet = df_brief, columns = ['Total', 'Date']

df2 = df_brief.copy()
df2 = df2.sort_values(by=['Year', 'Month #'])
#df2 = df2.groupby(by=['Year'])['Total Departures Count', 'Month'].sum()
df2 = df2.groupby(['Year', 'Month #', 'Month']).sum().reset_index()
fig = px.line(df2, x='Month', y='Total Departures Count', animation_frame='Year',
              range_y=[0,max(df2['Total Departures Count'])+10000])#, barmode='group')
fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 20 #stay
fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 10 #switch
st.plotly_chart(fig)
#make barcharts of years over the months ---

#########################   Visualization Number 3 ########################

st.subheader('Drill-down monthly trend chart of flight traffic for the top 15 countries')
#country wise, count tren over months, over years
#sheet = df_detailed
#plt.ylabel('Departures Count')
#plt.title("Drill-down trend chart for months from Januray to March for the top 15 countries")

df3 = df_detailed.copy()


user_n_chosen = st.selectbox(
   "Choose the number of countries you want to analyze",
    ('Top 5', 'Top 10', 'Top 15','Top 20'),
    index=2)


frame_duration = 150
transition_duration = 90

top_n = int(user_n_chosen.split()[-1]) #can be a user input
top_n_countries = list(df3[df3['Year'] == 2019].groupby('Destination_Airport_Country', sort = True).sum()['Total Departures Count'].nlargest(top_n).index)

def convert_datetime(dt):
    return datetime.strftime(dt, '%Y-%m-%d')

#index_names = df3[df3['Year'] <2019].index
#df3.drop(index_names, inplace = True)
df3 = df3.groupby(['Year','Month #','Date', 'Destination_Airport_Country']).sum().reset_index()
df3 = df3[df3['Destination_Airport_Country'].isin(top_n_countries)]
df3['Date'] = df3['Date'].apply(convert_datetime)
fig3 = px.bar(df3,
               x = 'Destination_Airport_Country',
               y = 'Total Departures Count',
               color = 'Destination_Airport_Country',
               animation_frame='Date')
fig3.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 150 #stay
fig3.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 90 #switch

st.plotly_chart(fig3)

########################################## CLIMATE DATA #################################################

st.header('Climate Data')

image_climate = Image.open('climate.jpg')
st.image(image_climate, width = 400)


########################################## DATASET #################################################

st.title('Dataset')

st.header('Dataset Description')

st.header('Maintaining 4 C\'s')

########################################## ADVANCED ML #################################################

st.title('Advanced Machine Learning')

########################################## EDA #################################################

st.title('Exploratory Data Analysis')

