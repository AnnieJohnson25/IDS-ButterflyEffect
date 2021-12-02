# Required libraries
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
from datetime import datetime
from plotly.subplots import make_subplots
import calendar
import datetime as dt
from sklearn.linear_model import LinearRegression

# Setting the title
st.title('COVID-19 Out of the box viewpoints')

image = Image.open('Data/out-of-the-box.jpg')
st.image(image, caption='COVID-19 Out of the box viewpoints')
 
########################################## LOADING DATA #################################################

@st.cache
def load_data():
    month_dict = {1:"Jan", 2:"Feb", 3:"Mar", 4:"Apr", 5:"May", 6:"Jun", 7:"Jul", 8:"Aug", 9:"Sep", 10:"Oct",
                  11:"Nov", 12:"Dec"}
    df_brief = pd.read_csv("Data/FlightsDepartureCount.csv")
    df_brief['Date'] = pd.to_datetime(df_brief["Date"])
    df_brief['Year'] =  pd.DatetimeIndex(df_brief["Date"]).year
    df_brief['Month #'] = pd.DatetimeIndex(df_brief["Date"]).month
    df_brief['Month'] = df_brief["Month #"]
    df_brief = df_brief.replace({"Month": month_dict})

    df_detailed = pd.read_csv("Data/FlightsDepartureWithCountry.csv")
    df_detailed['Date'] = pd.to_datetime(df_detailed["Date"])
    df_detailed['Year'] = pd.DatetimeIndex(df_detailed["Date"]).year
    df_detailed['Month #'] = pd.DatetimeIndex(df_detailed["Date"]).month
    df_detailed['Month'] = df_brief["Month #"]
    df_detailed = df_detailed.replace({"Month": month_dict})

    return df_brief, df_detailed

df_brief, df_detailed = load_data()

# Side Navigation bar
st.sidebar.markdown("# [About] (#about)")
st.sidebar.markdown("# [Visualizations] (#visualizations)")
st.sidebar.markdown("# [Dataset](#dataset)")
st.sidebar.markdown("# [Advanced Machine Learning](#advanced-machine-learning)")
st.sidebar.markdown("# [Exploratory Data Analysis](#exploratory-data-analysis)")

########################################## ABOUT #################################################

st.title('About')

st.header('Problem Statement')

st.write("""
This project focuses on comparing some overlooked parameters and metrics that changed for good during the pandemic. This project specifically focuses on
""")

st.write("""
1. Gaming trend
2. Flight travel and carbon emissions
3. Suicide rate in the United States of America
4. Climate change across the globe
""")

st.header('Demo')
st.video('https://youtu.be/fDek6cYijxI')

st.header('Project Team')
st.write("""
#### 1. Abhishree Shetty
#### 2. Sundar Anand
#### 3. Annie Johnson
#### 4. Jayant
""")

st.title('Visualizations')

########################################## COVID DATA #################################################

# Introduction
st.header('Interesting trends when the pandemic happened - COVID 19')

# Reading the file
country_df = pd.read_csv("Data/covid_country_wise.csv")
country_df['date'] = pd.to_datetime(country_df['date'], format='%d/%m/%y').dt.date

# Getting only vaccination data
vaccine_df = country_df[(country_df['percentage_vaccinated'].notna()) & (country_df['percentage_active'].notna())]
vaccine_df = vaccine_df.sort_values(by='date')

# Getting the country list
country_list = sorted(vaccine_df['country'].unique())

st.write("Let's compare the vaccination time-series data between different countries in the same time period...")

# Country wise comparison bar graphs
countries = st.multiselect("Which countries do you like to see?", country_list, ['United States', 'India', 'Brazil'])
date = st.selectbox("Pick a date", options = ['autoplay'] + list(vaccine_df[vaccine_df['country'].isin(countries)]['date'].unique()))

# For a selected date
if date != 'autoplay':
    conditioned_df = vaccine_df[(vaccine_df['country'].isin(countries)) & (vaccine_df['date'] == pd.to_datetime(date))]
    fig = px.bar(conditioned_df, x='country', y=['percentage_vaccinated', 'percentage_active'], barmode='group')

# For all dates
else:

    # Getting only the countries and sorting it by time and country
    conditioned_df = vaccine_df[(vaccine_df['country'].isin(countries))]
    conditioned_df['date'] = pd.to_datetime(conditioned_df['date'])
    conditioned_df = conditioned_df.sort_values(by=['date'])
    conditioned_df['date'] = conditioned_df['date'].dt.strftime('%Y-%m-%d')
    conditioned_df = conditioned_df.sort_values(by=['percentage_vaccinated', 'date', 'country'])

    # Plotting the graph and animating it
    fig = px.bar(conditioned_df, x='country', y=['percentage_active', 'percentage_vaccinated'], animation_frame='date', barmode='group')

    # Increasing the animation speed
    fig.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 10
    fig.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 4

# Displaying the graph
fig.update_layout(width=800)
st.plotly_chart(fig, use_container_width=False)

st.write('Take turns to see the vaccination percentage and active case count across countries on different dates')

# Getting country and date as input
default = country_list.index('United States')
option = st.selectbox('Pick a country', country_list, index=default)
country_vaccine_df = vaccine_df[(vaccine_df['country'] == option)]
date = st.select_slider("Pick a date", options = country_vaccine_df[country_vaccine_df['active_cases'].notna()]['date'].unique())

# Calculating the percentage of vaccination

vacc_info_df = country_vaccine_df[(country_vaccine_df['date'] == pd.to_datetime(str(date)))]
vacc = vacc_info_df['people_vaccinated'].iloc[0]
per = round(vacc_info_df['percentage_vaccinated'].iloc[0], 2)
count = country_df[(country_df['country'] == option) & (country_df['date'] == pd.to_datetime(str(date)))]['active_cases'].iloc[0]

# Displaying the results
st.write("The total number of vaccinated people are " + str(vacc) + ", the vaccination percentage as of " + str(date) + " is " + str(per) + "%, yet the active cases count is " + str(count))

# Vaccination vs Active rate graph for multiple countries
countries = st.multiselect("Which countries would you like to see?", country_list, ['United States', 'India', 'Brazil', 'Germany', 'United Kingdom', 'Russia', 'France', 'Malaysia', 'Japan'])
vaccine_df['date'] = pd.to_datetime(vaccine_df['date']).dt.strftime('%Y-%m-%d')
vaccine_df = vaccine_df[(vaccine_df['country'].isin(countries))]

# Plotting the graph
figcovid = px.scatter(vaccine_df, x="percentage_vaccinated", y="percentage_active", animation_frame="date", 
    animation_group="country", range_x=[-5,max(vaccine_df['percentage_vaccinated'])], range_y=[-1,max(vaccine_df['percentage_active'])], color='country', size='population')

# Increasing the animation speed
figcovid.layout.updatemenus[0].buttons[0].args[1]['frame']['duration'] = 30
figcovid.layout.updatemenus[0].buttons[0].args[1]['transition']['duration'] = 15

# Displaying the graph
figcovid.update_layout(width=800)
st.plotly_chart(figcovid, use_container_width=False)

st.subheader('Sad times. But how did other mental and environmental things turn up? Was there any segment that got beaten up or flourished in this pandemic?')

########################################## GAMING DATA #################################################

st.header('Gaming Data')

image_gaming = Image.open('Data/gaming.jpg')
st.image(image_gaming, width = 400)

# Reading the data
twitch_df=pd.read_csv('Data/Twitch_global_data.csv')
covid_df=pd.read_csv('Data/covid_global.csv')

#########################   Visualization Number 1 ########################

st.subheader('Comparing the number of hours globally spent on watching twitch streams every month from 2016 to 2021 September')

# Concatenating year and month columns into month-year column
twitch_df["month-year"]=twitch_df["Month"].astype(str)+"-"+twitch_df["year"].astype(str)
# Converting the string month-year to datetime
twitch_df["month-year"]=pd.to_datetime(twitch_df["month-year"])
# Finding number of millions of hours watched and storing this data in column Hours_watched_millions
twitch_df["Hours_watched_millions"]=twitch_df["Hours_watched"]/1000000
# Plotting the Hours_watched_millions data Vs month-year

# Creating month, year and full_month columns in the twitch_df dataframe
twitch_df['month'] = pd.DatetimeIndex(twitch_df['month-year']).month
twitch_df['year'] = pd.DatetimeIndex(twitch_df['month-year']).year
twitch_df['full_month'] = twitch_df['month'].apply(lambda x: calendar.month_abbr[x])

# Making an animated bar plot depicting the number of hours of watched in millions Vs each month from 2016-2021 September
fig = px.bar(twitch_df, x="full_month", y="Hours_watched_millions", color="month",
  animation_frame="year", animation_group="full_month", range_y=[0,3000])

st.write(fig)


#########################   Visualization Number 2 ########################

st.subheader('Comparing the rise in COVID-19 active cases along with the average and peak number of twitch stream viewers from 2016')

# Finding average viewers in thousands and storing this data in Avg_viewers_thousands
twitch_df["Avg_viewers_thousands"]=twitch_df["Avg_viewers"]/1000
# Finding number of peak viewers in thousands and storing this data in column Peak_viewers_thousands
twitch_df["Peak_viewers_thousands"]=twitch_df["Peak_viewers"]/1000
# Calculating streams in millions and storing in streams_millions
twitch_df["Streams_millions"]=twitch_df["Streams"]/1000000
# Storing dates in datetime format in column date
twitch_df["date"]=pd.to_datetime(twitch_df["month-year"])
twitch_df["month-year"]=pd.to_datetime(twitch_df["month-year"]).dt.to_period('m')

# Preprocessing the covid_df data
covid_df["date"]=pd.to_datetime(covid_df["date"],format="%d/%m/%y")
covid_df["month-year"]=covid_df["date"].dt.to_period('m')
covid_df=covid_df.drop_duplicates(subset=['month-year'], keep='first')

# Performing left join on twitch_df and covid_df to get all the rows having only dates mentioned in twitch_df
df=twitch_df.merge(covid_df, how='left', on=['month-year'])

# Create plot

fig = make_subplots(specs=[[{"secondary_y": True}]])

fig.add_trace(
    go.Scatter(x=list(df["date_x"]), y=list(df.Avg_viewers_thousands), name="Average number of viewers in thousands"))

fig.add_trace(
    go.Scatter(x=list(df["date_x"]), y=list(df.Peak_viewers_thousands), name="Peak number of viewers in thousands"))

fig.add_trace(
    go.Scatter(x=list(df["date_y"]), y=list(df["active_cases"]), name="Active cases per day"), secondary_y=True)

# Labelling both axis
fig.layout.xaxis.title="Year"

fig.layout.yaxis.title="Viewers in thousands"

fig.update_yaxes(title_text="Number of active cases", secondary_y=True)

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.15,
    xanchor="right",
    x=1
))

# Add range slider
fig.update_layout(
    xaxis=dict(
        rangeselector=dict(
            buttons=list([
                dict(count=1,
                     label="1m",
                     step="month",
                     stepmode="backward"),
                dict(count=6,
                     label="6m",
                     step="month",
                     stepmode="backward"),
                dict(count=1,
                     label="YTD",
                     step="year",
                     stepmode="todate"),
                dict(count=1,
                     label="1y",
                     step="year",
                     stepmode="backward"),
                dict(step="all")
            ])
        ),
        rangeslider=dict(
            visible=True
        ),
        type="date"
    )
)

st.write(fig)

#########################   Visualization Number 3 ########################

st.subheader('Comparing the number of streams in millions alongside the number of COVID-19 active cases from 2016')

# Making the plot

fig = make_subplots(specs=[[{"secondary_y": True}]])
 
fig.add_trace(
    go.Scatter(x=list(df["date_x"]), y=list(twitch_df.Streams_millions), name="Streams per month in millions"))

fig.add_trace(
    go.Scatter(x=list(df["date_y"]), y=list(df["active_cases"]), name="Active cases per day"), secondary_y=True,)


fig.layout.xaxis.title="Year"

fig.layout.yaxis.title="Number of streams in millions"

fig.update_yaxes(title_text="Number of active cases", secondary_y=True)

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.01,
    xanchor="right",
    x=1
))

st.write(fig)

########################################## SUICIDE DATA #################################################

st.header('Suicide Data')

image_suicide = Image.open('Data/suicide.jpeg')
st.image(image_suicide, width = 400)

### Header
st.subheader('US Suicide counts trend for years 2017 - 2020')

# Reading the suicide df
suicide_df = pd.read_csv("Data/suicide_cleaned.csv")

# Multi-select to choose the categories to plot
st.write('Category wise suicide rate per 100K pop alongside COVID numbers')
category = st.multiselect("Pick a categories", list(suicide_df.columns), ['Rate Age 45-54', 'Rate Sex Female', 'Rate Age 55-64'])

# Melting the dataframe to plot multiple lines
suicide_melt_df = pd.melt(suicide_df, id_vars=['Date', 'COVID count'], value_vars=category)
suicide_melt_df = suicide_melt_df.dropna()

# First plot Data vs Suicide rate
fig1 = px.line(suicide_melt_df, x="Date", y="value", range_y=[min(suicide_melt_df['value']), max(suicide_melt_df['value'])], color='variable')

# Secondary plot Data vs Covid rate
fig2 = px.bar(suicide_df, x='Date', y='COVID count', range_y=[0, max(suicide_melt_df['COVID count'])])
fig2.update_traces(yaxis="y2", opacity=0.6)

# Making Subplots
subfig = make_subplots(specs=[[{"secondary_y": True}]])

# Naming the axes
subfig.add_traces(fig1.data + fig2.data)
subfig.layout.xaxis.title="Date"
subfig.layout.xaxis.showgrid = False
subfig.layout.yaxis.title="Suicide / 100K pop"
subfig.layout.yaxis.showgrid = False
subfig.layout.yaxis2.title="COVID Rate value"
subfig.layout.yaxis2.showgrid = False

# Displaying the chart
st.write(subfig)

# Comparing year wise data for the chosen categories
st.write('Year wise suicide count for chosen categories')
fig = px.bar(suicide_melt_df, x='variable', y='value', color='Date', barmode='group')

# Displaying the chart
st.write(fig)

st.write('Looks like that the Suicide rate in most categories have reduced during the pandemic. Yet another good trend during the sad COVID time!!.')

########################################## FLIGHTS DATA #################################################

st.header('Flights Data')

image_flights = Image.open('Data/flights.jpg')
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

image_climate = Image.open('Data/climate.jpg')
st.image(image_climate, width = 400)

########################################## CONCLUSION #################################################
st.subheader("""
Seems like there were quite a few good things that happened during the pandemic which were overlooked.
Good that we were able to catch a few of them here!!
""")

########################################## DATASET #################################################

st.title('Dataset')

st.header('Dataset Description')

##### COVID Dataset #####

##### Gaming Dataset #####

##### Suicide Dataset #####

st.subheader('Suicide Rates - Data Source')

st.write("""
https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016
The dataset contains features respresenting the number of suicides per 100K population for each quater from 2017.
The data in the dataset is categorised acoording to both age and gender criteria.
""")

st.subheader('Problems faced in using this dataset')
st.write("""
- There is no authorised data source that authenticates global suicide rate.
- Few false positives and negatives in the dataset. Few natural death and accidents are reported as suicide and vice versa.
""")

st.subheader('Preprocessing done')
st.write("""
- Removed some unwanted columns
- Removed duplicates
- Typecased the data according to the format needed
- Reformatted and standardised the datetime
""")

##### Flight Dataset #####

##### Climate Dataset #####

st.header('Maintaining 4 C\'s')

########################################## ADVANCED ML #################################################

st.title('Advanced Machine Learning')

st.subheader("What if COVID never happened? Would the trend be any different? Let's find that using Multivariate Polynomial Regression")
st.subheader('The number of twitch stream watch hours predicted Vs the actual number of hours spent watching twitch during the COVID-19 pandemic')

# Preprocessing
date_object = dt.date(2020,1,1)
df["timestamp"]=df["date_x"].apply(lambda x: x.value)
df3 = df[df["date_x"].dt.date<date_object]
df3["timestamp"]=df3["date_x"].apply(lambda x: x.value)
df4 = df[df["date_x"].dt.date>=date_object]
df4["timestamp"]=df4["date_x"].apply(lambda x: x.value)
X = df3[["timestamp"]]
y = df3.loc[:, 'Hours_watched']  # create the target

# Fitting linear regresion model on timestamp data
model = LinearRegression()
model.fit(X, y)

# Test data
x_test=df4[["timestamp"]]
y_pred = model.predict(x_test)
y_test = df["Hours_watched_millions"].tolist()

plot_y=pd.concat([df3["Hours_watched_millions"],pd.Series(y_pred/1000000)])

print(plot_y)

# Plotting predicted Vs actual watch hours during the pandemic
fig = go.Figure()

fig.add_trace(
    go.Scatter(x=df["date_x"].tolist(), y=plot_y.tolist(), name="Forecasted hours watched in millions", opacity=0.5))

fig.add_trace(
    go.Scatter(x=df["date_x"].tolist(), y=y_test, name="Actual hours watched in millions", opacity=0.5))

# Labelling the axis
fig.layout.xaxis.title="Year"

fig.layout.yaxis.title="Hours watched in millions"

fig.update_layout(legend=dict(
    orientation="h",
    yanchor="bottom",
    y=1.01,
    xanchor="right",
    x=1
))

st.write(fig)

########################################## EDA #################################################

st.title('Exploratory Data Analysis')

st.write("The complete EDA notebook for this project can be found [here](https://github.com/abhishreeshetty/IDS-ButterflyEffect/blob/main/FinalProject_EDA.ipynb)")

# Images
images = ['Data/gaming_eda.png', 'Data/flights_eda.png', 'Data/suicide_eda.png', 'Data/climate_eda.png']
st.image(images, use_column_width=True)

st.header('Thank you!!')