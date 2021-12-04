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
import altair as alt
from vega_datasets import data
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from dateutil.relativedelta import relativedelta

# Setting the title
st.title('COVID-19 Out of the box viewpoints')

image = Image.open('Data/out-of-the-box.jpg')
st.image(image, caption='COVID-19 Out of the box viewpoints')
 
########################################## LOADING DATA #################################################

@st.cache()
def getDatasets():
    df2019 = pd.read_csv("Data/pollutionData2019.csv", encoding = "UTF-8")
    df2020 = pd.read_csv("Data/pollutionData2020.csv", encoding = "UTF-8")
    df2021 = pd.read_csv("Data/pollutionData2021.csv", encoding = "UTF-8")

    df_with_country_name = pd.concat([df2019, df2020, df2021])

    data_species_list = df_with_country_name['Specie'].unique()

    datasets = {}
    for specie in data_species_list:
        datasets[specie] = df_with_country_name[df_with_country_name['Specie'] == specie]
    
    return datasets

@st.cache()
def getCountryCodes():
    df_with_country_code = pd.read_csv("Data/wikipedia-iso-country-codes.csv", encoding = "UTF-8")
    return df_with_country_code

@st.cache()
def getGeographyData(locationData, specieFilter):
    locationData = locationData.pivot_table(index = "Date", columns = 'Country Code', values = 'Value').reset_index().astype({"Date": "datetime64"}).set_index('Date')
    locationData = expSmooth(locationData).reset_index()
    dates_data = locationData
    dates_data = dates_data.melt(id_vars='Date', var_name='Country',
                value_name=specieFilter)

    country_codes = getCountryCodes()
    dates_data = dates_data.merge(country_codes, left_on='Country', right_on='Alpha-2 code')[["Date", "Numeric code", "English short name lower case", f"{specieFilter}"]].rename(
        columns = {
            "English short name lower case" : "Country",
            "Numeric code": "id"
        }
    )
    return dates_data

def difference(dataset, interval=1):
	diff = list()
	for i in range(interval, len(dataset)):
		value = dataset[i] - dataset[i - interval]
		diff.append(value)
	return np.array(diff)

# invert differenced value
def inverse_difference(history, yhat, interval=1):
	return yhat + history[-interval]

def getForecast(dataset, specie, cityOrCountryFilter, location, date):
    locationData = dataset[dataset[cityOrCountryFilter] == location][['Date', 'Value']]
    locationData = locationData.groupby(['Date']).aggregate('mean').reset_index()
    smoothValues = expSmooth(locationData['Value'])
    values = smoothValues.values
    dates = locationData['Date'].values
    initial_dates = [datetime.strptime(x, "%Y-%m-%d") for x in dates]
    data_end_date = initial_dates[-1]

    days_in_year = 365
    differenced = difference(values, days_in_year)

    # fit model
    model = ARIMA(differenced)
    model_fit = model.fit()

    steps = abs(date - data_end_date).days

    forecast = model_fit.forecast(steps=steps)

    history = [x for x in values]

    for yhat in forecast:
        inverted = inverse_difference(history, yhat, days_in_year)
        history.append(inverted)
    
    date_range = pd.date_range(start=data_end_date + relativedelta(days=1),end=date).to_pydatetime().tolist()

    # train global data
    globalData = dataset[['Date', 'Value']]
    globalData = globalData.groupby(['Date']).aggregate('mean').reset_index()
    globalSmoothValues = expSmooth(globalData['Value'])
    globalValues = globalSmoothValues.values
    
    global_differences = difference(globalValues, days_in_year)
    # fit model
    global_model = ARIMA(global_differences)
    global_model_fit = model.fit()

    global_forecast = global_model_fit.forecast(steps=steps)
    global_history = [x for x in globalValues]

    for yhat in global_forecast:
        inverted = inverse_difference(global_history, yhat, days_in_year)
        global_history.append(inverted)

    chart = pd.DataFrame({'Date': list(date_range), f'{location}': history[-len(date_range):], f'Global Average': global_history[-len(date_range):]}).set_index('Date')
    st.subheader(f'Forecast for {specie} for one year (in PPM)')

    last_forecast = history[-1]
    locationData['Date'] = pd.to_datetime(locationData['Date'])

    #st.metric(f"{specie} (in PPM)", f"{last_forecast.round(2)}")

    st.line_chart(chart)

def plotCloropleth(dates_data, specieFilter, date, background):
    dates_data = dates_data[dates_data['Date'] == date]

    source = alt.topo_feature(data.world_110m.url, "countries")
    date_str = date.strftime("%d %B %Y")
    

    foreground = (
        alt.Chart(source, title=f"Geographic distribution of {specieFilter} in PPM as of {date_str}")
        .mark_geoshape(stroke="black", strokeWidth=0.15)
        .encode(
            color=alt.Color(
                f"{specieFilter}:Q", scale=alt.Scale(scheme="yelloworangered"),
            ),
            tooltip=[
                alt.Tooltip("Country:N", title="Country"),
                alt.Tooltip(f"{specieFilter}:Q", title=f"{specieFilter} in PPM"),
            ],
        )
        .transform_lookup(
            lookup="id",
            from_=alt.LookupData(dates_data, "id", [f"{specieFilter}", "Country"]),
        )
    )

    final_map = (
        (background + foreground)
        .configure_view(strokeWidth=0)
        .properties(width=700, height=400)
        .project("naturalEarth1")
    )

    return final_map

def reportTrends(locationData, regionCriteria):
    st.subheader("Pre-covid vs covid times")
    data_cols = list(locationData.columns)
    num_cols = len(data_cols)
    cols = st.columns(num_cols)
    locationData = locationData.reset_index()

    covid_clock = datetime(2020, 1, 1)

    for i in range(num_cols):
        location = data_cols[i]
        pre_covid_mean = locationData[locationData['Date'] < covid_clock][location].mean()
        post_covid_mean = locationData[locationData['Date'] >= covid_clock][location].mean()
        percentage_change = 100 * (post_covid_mean - pre_covid_mean)/pre_covid_mean
        cols[i].metric(f"{location} (in PPM)", f"{post_covid_mean.round(2)}", f"{percentage_change.round(2)}%")

def expSmooth(series, alpha = 0.02):
    return series.ewm(alpha=alpha, ignore_na = True).mean()

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

st.subheader('The butterfly effect')

st.write("""
With the advances in science in recent times, it feels like we are advancing towards a society where we have enough resources to predict everything that can happen and maybe also explain how and why of it. For example, we can make accurate hourly weather forecasts up to a few days in the future. Feats like this were not even dreamt of a few centuries ago. This success is also seen in looking into the past. Science enables us to solve uncountable mysteries of the past on a daily basis (yet many more seem to pop up every day). 

Given all this, it is possible to think of the future as deterministic and as something we just have to wait for to manifest itself. But here is the catch - every system is deterministic only if you know the current state of the system to infinite precision. This is called sensitive dependence on initial conditions, i.e, any slight difference in the current state could lead us to a completely different future. Why is this the case? This is a characteristic of a chaotic system. The classic examples of a chaotic system are: two-body pendulum and water flow in a pipe. But these are not the only chaotic systems. The more and more we study different systems, we realize that most of the worldly systems are chaotic. Even the earth going around the sun is a chaotic system.

What does it mean to be in a chaotic system? We can never predict anything beyond a small time frame in the future to a decent precision. Take the example of the weather forecast systems. The reason the forecasts are limited to a week in the future is because any prediction we might make beyond that would be just bad. Weather being chaotic in nature, a butterfly’s flapping in Mexico could cause a Hurricane in Florida. Anything could effect anything.

This brings us to the question: Then, how do we determine the chain of causality in a system? The simple answer is: we can’t. All we can do is use domain knowledge and data to give most probable reasons for an event. One might argue that causality could be determined in a Randomized Controlled Test. This is not possible as we would require infinite trails, perfect data (in terms of precision), and zero errors to make any claims of perfect determinism (This is the reason we report significance of our results in an RCT).
""")

st.subheader('Illustration of a Chaotic System')

st.write("""
The world we live in today is a perfectly relatable example of a chaotic system. For even the biggest of events happening in the society, we cannot determine what its repercussions would be. Note that we are flipping the perspective from “What caused this event?” to “What will this event cause?”. 

Let us pick the biggest event that happened in the recent years: Outbreak of the Covid-19 virus. Specifically, we want to look at its correlations with Air Travel, Game Streaming, Air Pollution, Suicide rates. We have put them in the decreasing order of our ability to think of a chain of causality for it.
""")

st.subheader('Inspiration')
st.write("""
Our experiment and explanation were inspired from a Youtube video by Veritasium - Chaos: The Science of the Butterfly Effect - YouTube.
""")

st.video('https://youtu.be/fDek6cYijxI')

st.write("""
This project focuses on comparing some overlooked parameters and metrics that changed for good during the pandemic. This project specifically focuses on
""")

st.write("""
1. Gaming trend
2. Flight travel and carbon emissions
3. Suicide rate in the United States of America
4. Climate change across the globe
""")

st.write("""
#### Potential Research Questions?
1. Do trends in the COVID data have correlations to aspects of the world like Flights Traffic, Climate, Gaming that could've been overlooked by media?
2. Can COVID data be correlated to the trends in Suicide data? If so, did it impact differently for different age groups?
3. What role did video games play during the pandemic? 
4. Do the trends in flight traffic from U.S. to other countries during COVID tell us something new? Is the flight traffic reduction directly correlated to the trend in the COVID cases for that country?
5. Did the restrictions posed by COVID help combat climate issue? If so, how does the trend look like for different countries?
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

st.write("""
From the streaming data, we can see that game streaming has increased significantly since 2020. We immediately relate this to people being indoors more often since the outbreak. But then, we think of other causes too - what about increase in unemployment?, or just that this Amazon subsidiary started to market itself aggressively due to its competition from Microsoft. The chain of causality is getting a little difficult to come up with.
""")

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

st.write("""
This case shows how suicide rates have gone down since covid has started. If you start to reason about its relation and causal chaining with covid, there are multiple mindboggling factors that you could talk about for hours and still arrive on no conclusions. Most people would stop trying to analyze the causality and just say that is coincidental.
""")

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

st.write("""
The data clearly shows that Air traffic has reduced significantly after the covid outbreak. The data merely shows a correlation between these two events. After observing the correlation, it is the observer who forms a chain of causality in the mind that because of the worldwide travel restrictions that are caused by covid, there is a reduced air travel. Yes, this seems to be right. But that is not what the data tells us. We are the ones who performed this inference.
""")

########################################## CLIMATE DATA #################################################

st.header('Climate Data')

image_climate = Image.open('Data/climate.jpg')
st.image(image_climate, width = 400)

with st.spinner('Bringing you awesome...'):
    datasets = getDatasets()

species_of_interest = ['PM10', 'SO2','PM25', 'NO2', 'CO']

st.title('Pollution trends')

specieFilter = st.selectbox('Type of Pollutant', species_of_interest)
if specieFilter is None:
    specieFilter = 'PM25'
datasetOfInterest = datasets[specieFilter]

st.header('Time Series trends')

countries_list = datasetOfInterest['Country'].unique()
cities_list = datasetOfInterest['City'].unique()

cityOrCountryFilter = st.selectbox('Filter Criteria: City or Country',['City', 'Country'])
if cityOrCountryFilter is None:
    cityOrCountryFilter = 'City'

listOfPlaces = cities_list if cityOrCountryFilter == 'City' else countries_list

placeFilter = st.multiselect('Location filter - Choose the locations for which you want to see the trends', listOfPlaces)

plural = ''
if cityOrCountryFilter == 'Country':
    plural = 'countries'
elif cityOrCountryFilter == 'City':
    plural = 'cities'

globalAggregate = datasetOfInterest[['Date','Value']].groupby('Date').aggregate('median').reset_index().astype({"Date": "datetime64"}).set_index('Date')
globalAggregate['Value'] = expSmooth(globalAggregate['Value'])
globalAggregate.rename(columns={'Value': 'Global Average'}, inplace=True)

displayData = globalAggregate

if placeFilter and len(placeFilter)<5:
    locationData = datasetOfInterest[datasetOfInterest[cityOrCountryFilter].isin(placeFilter)][['Date', 'Value', cityOrCountryFilter]]
    locationData = locationData.pivot_table(index = "Date", columns = cityOrCountryFilter, values = 'Value').reset_index().astype({"Date": "datetime64"}).set_index('Date')
    locationData = expSmooth(locationData)
    displayData = globalAggregate.merge(locationData, on='Date')
elif len(placeFilter)>=5:
    st.warning(f"Select any number of {plural} between 1 and 4. There are {len(placeFilter)} places that are currently selected.")

st.line_chart(displayData)

reportTrends(displayData, cityOrCountryFilter)

st.header('Geographic trends')

world = data.world_110m.url

locationData = datasetOfInterest[['Date', 'Value', 'Country Code']]
dates_data = getGeographyData(locationData, specieFilter)

source = alt.topo_feature(data.world_110m.url, "countries")
background = alt.Chart(source).mark_geoshape(fill="white")

date_range = pd.date_range(start="2019-01-01",end="2021-11-20").to_pydatetime().tolist()
for i in range(len(date_range)):
    date_range[i] = date_range[i].date()

date = st.select_slider("Pick a date", options = date_range)
date = datetime(date.year, date.month, date.day)
final_map = plotCloropleth(dates_data, specieFilter, date, background)
plot = st.altair_chart(final_map)

st.write("""
For many pollutants, we see a global decline in the concentration in the atmosphere after the outbreak. This trend can be seen in many countries and cities as well. This can be attributed to many factories being shut down lately because of the pandemic. But this is not the only cause we can think of. Governments across the world are taking initiatives to reduce global warming. People are becoming more aware of the pollution caused by the meat industry and are opting to be vegan. Any of this (or all of them) can cause this betterment of the environment. At this point in our illustration, the causal chains formed in our head are becoming longer and harder to keep track of.
""")

########################################## CONCLUSION #################################################
st.subheader("Seems like there were quite a few good things that happened during the pandemic which were overlooked.")
st.write("""
Going by the illustrations, we can see that as it gets difficult to form causal explanations in our head, our explanations for the correlation goes from ‘This is why this happened’ to ‘I think it is just a coincidence’. In reality, both are not completely true. We cannot infer causality, but neither can we say that something is coincidental (as everything effects everything). So, when next time someone asks you a question about causality or coincidence of two events, you reply - “It is just chaos!”
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


####### GAMING DATA ########

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

####### CLIMATE DATA ########

with st.spinner('Bringing you awesome...'):
    datasets = getDatasets()

species_of_interest = ['PM10', 'SO2','PM25', 'NO2', 'CO']

st.title('Pollution Forecast')

st.write('We use ARIMA (Autoregressive Integrated Moving Average) model from Statsmodels package to forecast the trends in the pollutants. The seasonal changes in the pollutants are corrected for assuming a cycle of one year.')
st.markdown("[Statsmodels ARIMA documentation](https://www.statsmodels.org/stable/generated/statsmodels.tsa.arima.model.ARIMA.html#statsmodels.tsa.arima.model.ARIMA)")

st.subheader('Select the location for which you want the forecast')


countries = set()
cities = set()

for specie in species_of_interest:
    countries_ = set(datasets[specie]['Country'].unique())
    cities_ = set(datasets[specie]['City'].unique())

    if len(countries) == 0:
        countries = countries.union(countries_)
    else:
        countries = countries.intersection(countries_)
    
    if len(cities) == 0:
        cities = cities.union(cities_)
    else:
        cities = cities.intersection(cities_)

cities = list(cities)
countries = list(countries)

cityOrCountryFilter = st.selectbox('Filter Criteria: City/Country',['Country', 'City'])
if cityOrCountryFilter is None:
    cityOrCountryFilter = 'Country'

listOfPlaces = cities if cityOrCountryFilter == 'City' else countries

location = st.selectbox('Location name', listOfPlaces)

# date_range = pd.date_range(start="2021-12-01",end="2022-11-30").to_pydatetime().tolist()
# for i in range(len(date_range)):
#     date_range[i] = date_range[i].date()

# date = st.select_slider("Pick a date", options = date_range)
# date = datetime(date.year, date.month, date.day)

date = datetime(2022, 11, 20)

specie = st.selectbox("Pick the Pollutant", species_of_interest)

getForecast(datasets[specie], specie, cityOrCountryFilter, location, date)

########################################## EDA #################################################

st.title('Exploratory Data Analysis')

st.write("The complete EDA notebook for this project can be found [here](https://github.com/abhishreeshetty/IDS-ButterflyEffect/blob/main/FinalProject_EDA.ipynb)")

# Images
images = ['Data/gaming_eda.png', 'Data/flights_eda.png', 'Data/suicide_eda.png', 'Data/climate_eda.png']
st.image(images, use_column_width=True)

st.header('Thank you!!')