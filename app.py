# import libraries
import streamlit as st
import folium
import streamlit.components.v1 as stc
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from Airbnb import get_recommendations
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS
from streamlit_folium import folium_static
from folium.plugins import Fullscreen
from folium.plugins import HeatMap

# load the dataset function
def load_data(data):
    new_df = pd.read_csv('listings_cleaned.csv')
    return new_df

new_df = load_data('listings_cleaned.csv')

st.set_page_config(page_title="Airbswm Recommendation")
st.sidebar.image("Photos/logo1.png", width=200)
menu = ['Home','Recommend', 'About Airswm']


menu2 = sorted(new_df['neighbourhood_cleansed'].unique())
x1="All"
menu2 = np.insert(menu2,0,x1)


menu3 = ["Number of listings by neighbourhood","Number of listings by roomtype",
         "Property Type","Number of people per booking", "Average daily price",
         "Per neighbhood's Average review score in Istanbul"]

menu4 = new_df['room_type'].unique()
x2="All"
menu4 = np.insert(menu4,0,x2)


menu5 = new_df['accommodates']
# x3="All"
# menu5 = np.insert(menu5,0,x3)

choose_menu = st.sidebar.selectbox('Main Menu', menu)

name_corpus = "Şişli Beyoğlu Beşiktaş Kadıköy Airbnb Rent city center House Apartment Karaköy " \
              "neighborhoods Fatih Moda Cihangir cosmopolitan " \
              "My house is suitable for couples&business Pendik decorated apartment Enjoy a stylish experience " \
              "in our apartment Merkezî konumda bulunan bu sakin yerde" \
              "Cool decorated  house with private Pool Bu ferah ve benzersiz mekânda Home includes all you need in a home " \
              "Welcome to our modern and sweet 2-bedroom home İstiklal caddesine 2 dakika cati kati dairemiz arnavutkoyun en merkezi " \
              "Apartment is so close to Sabiha Gökçen Airport This house is a two room apartment"

coord = new_df.loc[:,['longitude','latitude']]

if choose_menu == 'Home':
    # st.write('<h1 style="text-align: center; ">AIRSWM</h1>', unsafe_allow_html=True)
    # st.write('<h2 style="font-size: 25px; text-align: center; ">ISTANBUL</h2>', unsafe_allow_html=True)
    # st.write('<h2 style="font-size: 25px; text-align: center; ">Stay With Me!</h2>', unsafe_allow_html=True)
    st.image("Photos/grafik ist.png", caption=None, width=None,  use_column_width=None, clamp=False, channels="RGB",
             output_format="auto")
    st.image("Photos/afiş1.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB",
             output_format="auto")
    st.image("Photos/afiş2.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB",
             output_format="auto")
    # st.write('This app is for recommending Airbnb listings in Istanbul. You can search for a listing and get the recommendation')
    # st.write('This app is created by Mert Yağcıoğlu')
    # st.subheader('WordCloud App')
    st.set_option("deprecation.showPyplotGlobalUse", False)
    name_wordcloud = WordCloud(stopwords=STOPWORDS, background_color='white', height=600, width=800).generate(
        name_corpus)
    plt.imshow(name_wordcloud)
    plt.axis('off')
    st.pyplot()

    #HeatMap
    st.subheader("Heat Map")
    map_folium = folium.Map([41.031, 28.990], zoom_start=11.3)
    Fullscreen().add_to(map_folium)
    HeatMap(new_df[['latitude', 'longitude']].dropna(), radius=8,
            gradient={0.2: 'blue', 0.4: 'purple', 0.6: 'orange', 1.0: 'red'}).add_to(map_folium)
    folium_static(map_folium , width=700, height=600)

    st.video("https://www.youtube.com/watch?v=-XSAqfK_UwY")

    #First 5 Rows
    st.subheader('Dataset screenshot of first 5 rows')
    st.write(new_df.head(5))

    choose_menu3 = st.sidebar.selectbox('Info as Graphs',menu3)
    if choose_menu3=="Number of listings by neighbourhood":
        # Seeing which neighborhood has the highest number of listings
        feq = new_df['neighbourhood_cleansed'].value_counts().sort_values(ascending=True)
        feq.plot.barh(figsize=(18, 12), color='b', width=1)
        plt.title('Number of listings by neighbourhood', fontsize=18)
        plt.xlabel('Number of listings', fontsize=14)
        plt.show(block=True)
        st.pyplot()

    # Unique Check
    new_df.room_type.unique()
    if choose_menu3 == "Number of listings by roomtype":
    # Histogram
        freq = new_df['room_type'].value_counts().sort_values(ascending=True)
        freq.plot.barh(figsize=(18, 12), width=1, color=['y', 'g', 'b', 'r'])
    # We give four different colours for our graphs
        plt.title("Number of listings by roomtype ", fontsize=20)
        plt.xlabel('Number of listings', fontsize=20)
        st.pyplot()

    if choose_menu3 == "Property Type":
        new_df.property_type.unique()

        # We want to Display the property types with at least 100 listings as there are too many different unique values
        # We will create a seperate columns 'total' to sort the data and then delete it once finished
        prop = new_df.groupby(['property_type', 'room_type']).room_type.count()
        prop = prop.unstack()
        # Returns a DataFrame having a new level of column labels whose inner-most level consists of the pivoted index labels.
        # To group the room_type together
        prop['total'] = prop.iloc[:, 0:3].sum(axis=1)
        # Create a column of to caculate the total
        prop = prop.sort_values(by=['total'])
        prop = prop[prop['total'] >= 100]
        # sort by ascending order to get the best and select only the groups of property types that have >=100 inputs
        prop = prop.drop(columns=['total'])
        # drop the 'Total' coumn once we are done

        # Histogram
        prop.plot(kind='barh', stacked=True, color=['r', 'g', 'b', 'y'],
              linewidth=1, grid=True, figsize=(18, 12), width=1)
        # Trying to match with the color pallete that we have above
        plt.title('Property types', fontsize=18)
        plt.xlabel('Number of listings', fontsize=18)
        plt.ylabel("")
        plt.legend(loc=4, prop={"size": 13})
        plt.rc('ytick', labelsize=13)
        st.pyplot()

    if choose_menu3 == "Number of people per booking":
        feq = new_df['accommodates'].value_counts().sort_index()
        feq.plot.bar(figsize=(18, 12), width=1, rot=0)
        plt.title('Number of people per booking in Seattle', fontsize=20)
        plt.ylabel('Number of listings', fontsize=18)
        plt.xlabel('Accommodates', fontsize=18)
        st.pyplot()

    if choose_menu3 == "Average daily price":
        # Histogram of Average daily price & Average review score by neighbourhood
        fig = plt.figure(figsize=(20, 10))
        plt.rc('xtick', labelsize=14)
        plt.rc('ytick', labelsize=15)
        ax1 = fig.add_subplot(121)
        feq = new_df[new_df['accommodates'] == 2]
        # take the property that accommodates 2 people only
        feq1 = feq.groupby('neighbourhood_cleansed')['price'].mean().sort_values(ascending=True)
        # groupby the 'neigborhood' then find the mean of the price and sort the value from ascedding order
        ax1 = feq1.plot.barh(color='r', width=1)
        plt.title("Average daily price for a 2-persons accommodation", fontsize=20)
        plt.xlabel('Average daily price ($)', fontsize=20)
        plt.ylabel("")
        st.pyplot()

    if choose_menu3 == "Per neighbhood's Average review score in Istanbul":
        fig = plt.figure(figsize=(20, 10))
        ax2 = fig.add_subplot(122)
        feq = new_df[new_df['number_of_reviews'] >= 10]
        # Take the lsitings with more than 10 reviews only
        feq2 = feq.groupby("neighbourhood_cleansed")['review_scores_location'].mean().sort_values(ascending=True)
        # groupby the 'neigborhood' then find the mean of the price and sort the value from ascedding order
        ax2 = feq2.plot.barh(color='b', width=1)
        plt.title("Per neighbhood's Average review score in Istanbul", fontsize=20)
        plt.xlabel('Score (scale 1-5)', fontsize=20)
        plt.ylabel("")
        plt.tight_layout()
        st.pyplot()

if choose_menu == 'Recommend':
    st.title('Istanbul Airswm Recommendation')
    st.image("Photos/logo2.png", caption=None, width=None, use_column_width=None, clamp=False, channels="RGB",
             output_format="auto")
    st.markdown("<h2 style='text-align: center; color: black;'>Recommendation</h2>",
                    unsafe_allow_html=True)
    choose_menu2 = st.sidebar.selectbox('Select Districts', menu2)
    choose_menu4 = st.sidebar.selectbox('Select Room Type', menu4)
    selected_id = st.number_input("Enter an Airswm ID:",format="%0f")
    if selected_id not in new_df['id'].tolist():
        st.write('<h2 style="font-size: 20px; text-align: justify; ">"Warning: The selected ID was not found in the data frame. Please enter a valid ID"</h2>',
                 unsafe_allow_html=True)
    else:
        recommendations = get_recommendations(selected_id)
        no_recommendation_flag = True
        if not recommendations.empty:
            st.write('<h2 style="font-size: 18px; ">"Recommended Listings:"</h2>',
                     unsafe_allow_html=True)
            for index, row in recommendations.iterrows():
                if choose_menu2=="All" and choose_menu4=="All":
                    longitude_list = []
                    latitude_list = []
                    district_list = get_recommendations(selected_id)["neighbourhood_cleansed"].tolist()
                    st.write("<p style='font-size: 18px;'>"f"Name: {row['name']}</p>", unsafe_allow_html=True)
                    st.write("<p style='font-size: 18px;'>"f"District: {row['neighbourhood_cleansed']}</p>", unsafe_allow_html=True)
                    st.write(f"Listing URL: {row['listing_url']}")
                    st.write("<p style='font-size: 18px;'>"f"Room Type: {row['room_type']}</p>",
                             unsafe_allow_html=True)
                    # longitude=st.write(f"latitude: {row['longitude']}")
                    longitude_list = get_recommendations(selected_id)['longitude'].tolist()
                    # latitude=st.write(f"longitude: {row['latitude']}")
                    latitude_list = get_recommendations(selected_id)['latitude'].tolist()
                    st.write("---")
                    no_recommendation_flag = False

                if row['neighbourhood_cleansed'] == choose_menu2  and choose_menu4=="All":
                    longitude_list = []
                    latitude_list = []
                    district_list = get_recommendations(selected_id)["neighbourhood_cleansed"].tolist()
                    st.write("<p style='font-size: 18px;'>"f"Name: {row['name']}</p>", unsafe_allow_html=True)
                    st.write("<p style='font-size: 18px;'>"f"District: {row['neighbourhood_cleansed']}</p>", unsafe_allow_html=True)
                    st.write(f"Listing URL: {row['listing_url']}")
                    st.write("<p style='font-size: 18px;'>"f"Room Type: {row['room_type']}</p>",
                             unsafe_allow_html=True)
                    # longitude=st.write(f"latitude: {row['longitude']}")
                    longitude_list = get_recommendations(selected_id)['longitude'].tolist()
                    # latitude=st.write(f"longitude: {row['latitude']}")
                    latitude_list = get_recommendations(selected_id)['latitude'].tolist()
                    st.write("---")
                    neighborhoods = recommendations['neighbourhood_cleansed'] == choose_menu2  # İlanların bulunduğu ilçeleri içeren sütun
                    ilce_satirlar = recommendations[neighborhoods]
                    longitude_list = []
                    latitude_list = []
                    longitude_list = ilce_satirlar["longitude"].tolist()
                    latitude_list = ilce_satirlar["latitude"].tolist()
                    no_recommendation_flag = False

                if choose_menu2 == "All" and row['room_type'] == choose_menu4:
                    longitude_list = []
                    latitude_list = []
                    district_list = get_recommendations(selected_id)["room_type"].tolist()
                    st.write("<p style='font-size: 18px;'>"f"Name: {row['name']}</p>", unsafe_allow_html=True)
                    st.write("<p style='font-size: 18px;'>"f"District: {row['neighbourhood_cleansed']}</p>", unsafe_allow_html=True)
                    st.write(f"Listing URL: {row['listing_url']}")
                    st.write("<p style='font-size: 18px;'>"f"Room Type: {row['room_type']}</p>",
                             unsafe_allow_html=True)
                    # longitude=st.write(f"latitude: {row['longitude']}")
                    longitude_list = get_recommendations(selected_id)['longitude'].tolist()
                    # latitude=st.write(f"longitude: {row['latitude']}")
                    latitude_list = get_recommendations(selected_id)['latitude'].tolist()
                    st.write("---")
                    neighborhoods = recommendations['room_type'] == choose_menu4  # İlanların bulunduğu ilçeleri içeren sütun
                    ilce_satirlar = recommendations[neighborhoods]
                    longitude_list = []
                    latitude_list = []
                    longitude_list = ilce_satirlar["longitude"].tolist()
                    latitude_list = ilce_satirlar["latitude"].tolist()
                    no_recommendation_flag = False

                # Seçilen ilçe ile tavsiye edilen ilçe aynıysa yazdır
                if row['neighbourhood_cleansed'] == choose_menu2 and row['room_type'] == choose_menu4:
                    longitude_list = []
                    latitude_list = []
                    district_list = get_recommendations(selected_id)["neighbourhood_cleansed"].tolist()
                    st.write("<p style='font-size: 18px;'>"f"Name: {row['name']}</p>", unsafe_allow_html=True)
                    st.write("<p style='font-size: 18px;'>"f"District: {row['neighbourhood_cleansed']}</p>", unsafe_allow_html=True)
                    st.write(f"Listing URL: {row['listing_url']}")
                    st.write("<p style='font-size: 18px;'>"f"Room Type: {row['room_type']}</p>",
                             unsafe_allow_html=True)
                    neighborhoods = recommendations['neighbourhood_cleansed'] ==choose_menu2  # İlanların bulunduğu ilçeleri içeren sütun
                    room_type = recommendations['room_type'] == choose_menu4
                    ilce_satirlar = recommendations[neighborhoods]
                    room_types = recommendations[room_type]
                    longitude_list = []
                    latitude_list = []
                    longitude_list1 = ilce_satirlar["longitude"].tolist()
                    latitude_list1 = ilce_satirlar["latitude"].tolist()
                    longitude_list2  = room_types["longitude"].tolist()
                    latitude_list2 = room_types["latitude"].tolist()
                    def intersection(longitude_list1 , longitude_list2):
                        return list(set(longitude_list1) & set(longitude_list2))
                    longitude_list  =(intersection(longitude_list1,longitude_list2))
                    def intersection(latitude_list1 , latitude_list2):
                        return list(set(latitude_list1) & set(latitude_list2))
                    latitude_list = (intersection(latitude_list1, latitude_list2))
                    st.write("---")
                    no_recommendation_flag = False

        if no_recommendation_flag:
            st.write("---")
            st.markdown("<h3 style='text-align: center; color: black;'>There is no recommendation!</h3>",
                        unsafe_allow_html=True)
            st.write("---")
            longitude_list = []
            latitude_list = []

        def main():
            st.markdown("<h2 style='text-align: center; color: black;'>Recommended Apartments</h2>",
                        unsafe_allow_html=True)

            # st.title("Önerilen Noktaları Haritada Gösterme")
            # Elde edilen enlem ve boylam değerlerini dizi olarak belirtin
            latitude_values = latitude_list
            longitude_values = longitude_list
            # Harita oluşturma
            m = folium.Map(location=[41.031, 28.990], zoom_start=11.3)
            Fullscreen().add_to(m)
            # Noktaları haritaya ekleme
            for lat, lon in zip(latitude_values, longitude_values):
                folium.Marker([lat, lon]).add_to(m)
                # Streamlit arayüzünde haritayı gösterme
            folium_static(m, width=800, height=600)

        if __name__ == "__main__":
            main()


if choose_menu == 'About Airswm':
    st.header("About us")
    st.write('<h2 style="font-size: 18px; text-align: justify; ">Airswm was born in 2023 when two Hosts welcomed three '
             'guests to their İstanbulo home, and has since grown to over 4 million Hosts who have '
             'welcomed over 1.5 billion guest arrivals in almost every country across the globe. Every day, '
             'Hosts offer unique stays and experiences that make '
             'it possible for guests to connect with communities in a more authentic way.</h2>', unsafe_allow_html=True)

    st.header("Latest news")

    st.write('<div style="display: flex; text-align: justify; font-size: 20px; '
             '"><img src="https://news.airbnb.com/wp-content/uploads/sites/4/2023/09/Llaes-Castle-Spain.jpeg?w=2048" '
             'width="450" height="375" style=" margin: 10px;'
             '"><a style="color: black"; href="https://news.airbnb.com/historical-homes-category-spreads-love-for-heritage-travel-across-europe/">'
             '‘Historical Homes’ category spreads love for heritage travel across Europe</a></div>',
        unsafe_allow_html=True)

    st.write('<div style="display: flex; text-align: justify; font-size: 20px; '
             '"><img src="https://news.airbnb.com/wp-content/uploads/sites/4/2021/04/MWL-Family-Friendly.jpeg?w=2048" '
             'width="450" height="375" style="margin: 10px;"><a style="color: black" href="https://news.airbnb.com/how-and-where-families-are-traveling-on-airbnb-this-summer/">'
             'How and where families are traveling on Airbnb this summer"</a></div>',
        unsafe_allow_html=True)

    st.write('<div style="display: flex; text-align: justify; font-size: 20px; '
             '"><img src="https://news.airbnb.com/wp-content/uploads/sites/4/2019/06/Checkin-1.png?w=1836" '
             'width="450" height="375" style="margin: 10px;'
             '"><a style="color: black" href="https://news.airbnb.com/an-update-on-our-work-to-crack-down-on-parties-and-disruptive-behavior/">'
             'An update on our work to crack down on parties and disruptive behavior</a></div>',
        unsafe_allow_html=True)

    st.write('<div style="display: flex; text-align: justify; font-size: 30px;font-weight: bold;'
             '"><a style="color: black";  href="https://www.rentalscaleup.com/airbnbs-2023-commercials-feature-new-categories-farms-islands-chefs-kitchen-islands-and-skiing-with-music-and-locations/">'
             'Airswm’s 2023 Commercials Feature New Categories</a></div>',unsafe_allow_html=True)
    st.subheader(" "
                 "")
    st.video("https://www.youtube.com/watch?v=EKaHDu5ebqg&t=11s")
    st.subheader(" "
                 "")
    st.subheader("Yellow Submarine | Airswm OMG! Category")
    st.write('<div style="display: flex; text-align: justify; font-size: 22px;'
             '"><a style="color: black";  href="https://www.airbnb.ca/rooms/20605023?source_impression_id=p3_1694114470_Iw6Ae5tn0l0NAeXx">'
             'Here’s the link to the Yello Submarine</a></div>',unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=5YuqQgRjG-U")
    st.subheader(" "
                 "")
    st.subheader("Old Town Road | Airswm Farms Category")
    st.write('<div style="display: flex; text-align: justify; font-size: 22px;'
             '"><a style="color: black";  href="https://www.airbnb.com.tr/rooms/12963071?_set_bev_on_new_domain=1693044159_YjZmN2QwZGZhNjg5&source_impression_id=p3_1694114795_e2et%2F5gFf1oynI3k">'
             'Here’s the link to the Coach House</a></div>',unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=lPtFZx0pv-8")
    st.subheader(" "
                 "")
    st.subheader("Voicemail | Airswm Islands Category")
    st.write('<div style="display: flex; text-align: justify; font-size: 22px;'
             '"><a style="color: black";  href="https://www.airbnb.com.tr/rooms/48008078?_set_bev_on_new_domain=1693044159_YjZmN2QwZGZhNjg5&source_impression_id=p3_1694114853_5q6Igvg8BxRFdNfe">'
             'Here’s the link to the Coach House</a></div>',unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=DdUKRnNbhvY")
    st.subheader(" "
                 "")
    st.subheader("Luna Mezzo Mare | Airswm Chef’s Kitchens Category")
    st.write('<div style="display: flex; text-align: justify; font-size: 22px;'
             '"><a style="color: black";  href="https://www.airbnb.co.uk/rooms/29642943?source_impression_id=p3_1694114917_%2BdLbwuMT6FruhfOK">'
             'Here’s the link to the Wensley</a></div>',unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=6kcp9aFLHVs")
    st.subheader(" "
                 "")
    st.subheader("Boombastic | Airswm Skiing Category")
    st.write('<div style="display: flex; text-align: justify; font-size: 22px;'
             '"><a style="color: black";  href="https://www.airbnb.com.tr/rooms/46338360?_set_bev_on_new_domain=1693044159_YjZmN2QwZGZhNjg5&source_impression_id=p3_1694114960_8pipt2FAzYiGDAQR">'
             'Here’s the link to the cabin in Corralco</a></div>',unsafe_allow_html=True)
    st.video("https://www.youtube.com/watch?v=vDv08msycQE")
    st.subheader(" "
                 "")
    # st.subheader('Recommending Airbnb listings in Istanbul')
    # st.write('This app is for recommending Airbnb listings in Istanbul. You can search for a listing and get the recommendation')
    st.write('<h2 style="font-size: 20px; text-align: justify; ">This app is created by</h2>', unsafe_allow_html=True)
    st.write('<h2 style="font-size: 15px; text-align: justify; ">Mert YAĞCIOĞLU</h2>', unsafe_allow_html=True)
    st.write('<h2 style="font-size: 15px; text-align: justify; ">Baki TURGUT</h2>', unsafe_allow_html=True)
    st.write('<h2 style="font-size: 15px; text-align: justify; ">Remzi CAN</h2>', unsafe_allow_html=True)
    st.write('<h2 style="font-size: 15px; text-align: justify; ">Mehmet EFENDİOĞLU </h2>', unsafe_allow_html=True)


get_recommendations(30697)




