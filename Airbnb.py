import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import warnings
warnings.filterwarnings('ignore')
import streamlit as st
import streamlit.components.v1 as stc
import folium
from streamlit_folium import folium_static
from folium.plugins import Fullscreen

from IPython.display import Image, HTML
from wordcloud import WordCloud, STOPWORDS
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import cosine_similarity
from numpy import *

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)
pd.set_option('display.max_rows', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)

data = pd.read_csv('listings.csv' , sep = "," , encoding = "utf-8")

data.head()
df2=data.copy()

df2.head(5)
# df2.info()
df2.shape
# df2.describe()
# df2.isnull().sum()

df2['name'] = df2['name'].astype('str')
df2['description'] = df2['description'].astype('str')

df3 = df2.loc[0:20000]
df3.head()
df3.shape
new_df = df3[["id", "name", "description", "host_name" ,"neighbourhood_cleansed" ,"listing_url","latitude",
              "longitude","room_type","property_type",'accommodates',"price",'number_of_reviews',
              'review_scores_location','accommodates']]
new_df.head(50)
def convert_price_to_int(price_str):
   try:
      price_str = price_str.replace('$', '').replace(',', '').replace('.00', '')  # Gereksiz karakterleri kaldırma
      return int(price_str)
   except ValueError:
      return None


new_df['price'] = new_df['price'].apply(convert_price_to_int)


new_df['content'] = new_df[['name', 'description']].astype(str).apply(lambda x: ' // '.join(x), axis = 1)
new_df['content'].fillna('Null', inplace = True)

tfidf = TfidfVectorizer(stop_words = 'english')
tfidf_matrix = tfidf.fit_transform(new_df["content"])
tfidf_matrix = tfidf_matrix.astype(np.float32)
cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
incides = pd.Series(new_df.index, index=new_df["id"])
incides.index.value_counts()
incides = incides[~incides.index.duplicated(keep="last")]
airbnb_index = incides[34566980]

similarity_scores = pd.DataFrame(cosine_sim[airbnb_index],columns=["score"])

airbnb_indices = similarity_scores.sort_values("score", ascending=False)[1:10].index

new_df['id'].iloc[airbnb_indices]

airbnb_id = 30697

def get_recommendations(airbnb_id):
   airbnb_idx = new_df[new_df['id'] == airbnb_id].index[0]
   sim_scores = list(enumerate(cosine_sim[airbnb_idx]))
   sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
   sim_scores = sim_scores[1:11]
   airbnb_indices = [i[0] for i in sim_scores]
   kolon = ["id","name", "description","listing_url","longitude","latitude","neighbourhood_cleansed","room_type",'accommodates']
   return new_df.loc[airbnb_indices,kolon]

kolon = ["id","name", "description","listing_url","longitude","latitude"]

# save the dataframe to csv
new_df.to_csv('listings_cleaned.csv', index=False)

# get_recommendations(30697)





