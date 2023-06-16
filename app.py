import streamlit as st
import pandas as pd
import numpy as np
import pickle 
# st.set_page_config(page_title="Crop Production", page_icon=":corn:")


st.set_page_config(layout="wide")

video_html = """
		<style>
		#myVideo {
		  position: fixed;
		  right: 0;
		  bottom: 0;
		  min-width: 100%; 
		  min-height: 100%;
		}
		.content {
		  position: fixed;
		  bottom: 0;
		  background: rgba(0, 0, 0, 0.5);
		  color: #f1f1f1;
		  width: 100%;
		  padding: 20px;
		}
		</style>	
		<video autoplay muted loop id="myVideo">
		  <source src="https://v1.pinimg.com/videos/mc/720p/8e/a5/4d/8ea54da9e5ff8d8c65c880463a2fb192.mp4")>
		  Your browser does not support HTML5 video.
		</video>
        """
st.markdown(video_html, unsafe_allow_html=True)

st.header ('Predict the Production of Crops at any Particular Season')

with open('mapping_dict.pkl','rb')as f:
    mapping_dict=pickle.load(f)

with open('model.pkl','rb')as f:
    model=pickle.load(f)

with open('data.pkl','rb')as f:
     data = pd.read_pickle(f)

def predict(State,District,Season,Area):
    state=mapping_dict['State'][State]
    district=mapping_dict['District'][District]
    season=mapping_dict['Season'][Season]


    prediction=model.predict(pd.DataFrame(np.array([state,district,season,Area]).reshape(1,4),columns=['State','District','Season','Area']))
    crop = list(filter(
        lambda x: mapping_dict['Crop'][x] == prediction, mapping_dict['Crop']))[0]
    return crop
    

# input 
#st.image("C:\Users\hc\Desktop\new_crop\pexels-pixabay-531880.jpg")
state_list =data['State'].unique()
selected_state=st.selectbox(
    "Select a state from the Dropdown",
    options=state_list
)

district_list =data['District'].unique()
selected_district=st.selectbox(
    "Select a district from the Dropdown",
    options=district_list
)
season_list=data['Season'].unique()
selected_season=st.selectbox(
    "Select a district from the Dropdown",
    options=season_list
)

Area = st.number_input('Area of plot in(Hectares)',min_value=0.00001,max_value=100000000.0,value=1.0)

if st.button('Predict Production'):
    st.subheader(predict(selected_state,selected_district,selected_season,Area))
    

css = """
h1 {
    color: blue;
    font-size: 36px;
    text-align: center;
}
p {
    color: white;
    font-size: 24px;
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
}
"""

# Render the CSS styles using st.markdown
st.markdown(f'<style>{css}</style>', unsafe_allow_html=True)


