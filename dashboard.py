import streamlit as st
import pandas as pd
import numpy as np
import os
import plotly.express as px
from wordcloud import WordCloud, STOPWORDS
import matplotlib.pyplot as plt
import base64
st.set_page_config( page_icon = "ben.png", page_title="Benjamin's Sentiment Analysis Project")
st.markdown('<link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">',
            unsafe_allow_html=True)

st.markdown(""" 
<nav class="navbar fixed-top navbar-expand-lg navbar-dark" style="background-color: #ff7f27">
  <a class="navbar-brand" href="https://www.aa.com" target="_blank">American Airlines</a>
  <button class="navbar-toggler" type="button" data-toggle="collapse" data-target="#navbarNav" aria-controls="navbarNav" aria-expanded="false" aria-label="Toggle navigation">
    <span class="navbar-toggler-icon"></span>
  </button>
  <div class="collapse navbar-collapse" id="navbarNav">
    <ul class="navbar-nav">
      <li class="nav-item active">
        <a class="nav-link disabled" href="#">Dashboard <span class="sr-only">(current)</span></a>
      </li>
      <li class="nav-item">
      <a class="nav-link" href="https://www.delta.com" target="_blank">Delta Air Lines</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://www.united.com" target="_blank">United Airlines</a>
      </li>
      <li class="nav-item">
        <a class="nav-link" href="https://www.southwest.com" target="_blank">Southwest Airlines</a>
      </li> 
    <li class="nav-item">     
  <a class="nav-link" href="https://www.virgin.com" target="_blank">Virgin America Airline</a><br><br>
      </li>
    </ul>
  </div>
</nav>

""", unsafe_allow_html=True)

#st.image("ben.png", width=40)
#here is the end of the code ................................
st.title("Benjamin's Internship Project Dashboard")
st.sidebar.markdown("### Dashboard & Analysis Corner")
#st.sidebar.title("Benjamin's Internship Project")
st.markdown(" This is a machine learning app used to analyze the sentiment of the tweets 🐦 about US airlines  ️ ")
st.sidebar.markdown("This app is a machine learning app used to analyze the sentiment of the tweets 🐦 about US airlines ✈️ ")
DATA_URL = ("Tweets.csv")
@st.cache(persist=True)
def load_data():
    data = pd.read_csv(DATA_URL)
    data['tweet_created'] = pd.to_datetime(data['tweet_created'])
    return data

data = load_data()
st.sidebar.subheader("Show random tweet")
random_tweet = st.sidebar.radio('Sentiment type', ('positive','negative','neutral'))
st.sidebar.markdown(data.query('airline_sentiment == @random_tweet')[["text"]].sample(n=1).iat[0,0])

st.sidebar.markdown("### Number of tweets by sentiment type")
select = st.sidebar.selectbox('Visualization type', ['Sentiment_Distribution', 'Pie_Chart',
                                                     'All_Airline_Sample', 'Sentiment_Per_Airline',
                                                     'Positive Sentiments', 'Negative Sentiments',
                                                     'Neutral Sentiments', 'Neg_Reasons_for_Neg_Senti'], key='1')
sentiment_count = data['airline_sentiment'].value_counts()
sentiment_count = pd.DataFrame({'Sentiment' :sentiment_count.index, 'Tweets' :sentiment_count.values})
airline_count=data['airline'].value_counts()
airline_count = pd.DataFrame({'airline' :airline_count.index, 'Tweets' :airline_count.values})
fig_group = data[data['airline_sentiment']=='positive'].airline.value_counts()
fig_group = pd.DataFrame({'airline_sentiment': fig_group.index, 'positive': fig_group.values})
fig_group_n = data[data['airline_sentiment']=='negative'].airline.value_counts()
fig_group_n = pd.DataFrame({'airline_sentiment': fig_group_n.index, 'negative': fig_group_n.values})
fig_group_nu = data[data['airline_sentiment']=='neutral'].airline.value_counts()
fig_group_nu = pd.DataFrame({'airline_sentiment': fig_group_nu.index, 'neutral': fig_group_nu.values})
negationsentiment=data['negativereason'].value_counts()
negationsentiment = pd.DataFrame({'negativereason' :negationsentiment.index, 'Tweets' :negationsentiment.values})



if not st.sidebar.checkbox('Hide', True):
    st.markdown("### Sentiment Analysis Outcome")
    if select == "Sentiment_Distribution":
        fig = px.bar(sentiment_count, x='Sentiment', y='Tweets', color = 'Tweets', height= 500)
        st.plotly_chart(fig)
    elif select=="Pie_Chart":
        fig = px.pie(sentiment_count, values='Tweets', names='Sentiment')
        st.plotly_chart(fig)
    elif select=="All_Airline_Sample":
        fig2=px.bar(airline_count, x='airline', y='Tweets', color='airline', height=500)
        st.plotly_chart(fig2)
    elif select=="Sentiment_Per_Airline":
        fg=pd.crosstab(data['airline'], data['airline_sentiment'])
        fig3=px.bar(fg)
        st.plotly_chart(fig3)
    elif select=="Positive Sentiments":
        fig_group_res=px.bar(fig_group, x='airline_sentiment', y='positive', color='airline_sentiment', height=500)
        st.plotly_chart(fig_group_res)
    elif select=="Negative Sentiments":
        fig_group_res2 = px.bar(fig_group_n, x='airline_sentiment', y='negative', color='airline_sentiment', height=500)
        st.plotly_chart(fig_group_res2)
    elif select=="Neutral Sentiments":
        fig_group_res3 = px.bar(fig_group_nu, x='airline_sentiment', y='neutral', color='airline_sentiment', height=500)
        st.plotly_chart(fig_group_res3)
    else:
        negationsentiment_res=px.bar(negationsentiment, x='negativereason', y='Tweets', color='Tweets', height=500)
        st.plotly_chart(negationsentiment_res)





st.sidebar.subheader("When and where are the users tweeting from?")
hour = st.sidebar.slider("Hour of day", 0, 23)
modified_data = data[data['tweet_created'].dt.hour == hour]
if not st.sidebar.checkbox("Close", True, key='1'):
    st.markdown("### Tweets location based on the time of the day")
    st.markdown("%i, tweets between %i:00 and %i:00" % (len(modified_data), hour, (hour+1)%24))
    st.map(modified_data)
    if st.sidebar.checkbox("Show raw data", False):
        st.write(modified_data)


st.sidebar.subheader("Breakdown airline tweets by sentiment")
choice = st.sidebar.multiselect("Pick airlines", ('US Airways', 'United', 'American', 'Southwest', 'Delta', 'Virgin America'), key = '0')


if len(choice) >0:
    choice_data = data[data.airline.isin(choice)]
    fig_0 = px.histogram(choice_data, x='airline', y='airline_sentiment', histfunc='count', color='airline_sentiment', facet_col = 'airline_sentiment', labels={'airline_sentiment':'tweets'}, height=600, width=800)
    st.plotly_chart(fig_0)

st.sidebar.header("Word Cloud")
word_sentiment = st.sidebar.radio('Display word cloud for which sentiment?', ('positive', 'negative', 'neutral'))

if not st.sidebar.checkbox("Close", True, key = '3'):
    st.header('Word cloud for %s sentiment' % (word_sentiment))
    df = data[data['airline_sentiment']==word_sentiment]
    words = ' '.join(df['text'])
    processed_words = ' '.join([word for word in words.split() if 'http' not in word and word.startswith('@') and word != 'RT'])
    wordcloud = WordCloud(stopwords=STOPWORDS, background_color = 'white', height=600, width=800).generate(processed_words)
    plt.imshow(wordcloud)
    plt.xticks([])
    plt.yticks([])
    st.pyplot()
    st.info("@ Benjamin Internship Project!")
