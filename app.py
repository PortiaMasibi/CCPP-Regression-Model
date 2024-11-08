# Here we will be using a Streamlit app

import streamlit as st
import pandas as pd
import numpy as np
from prediction import predict 


st.title("Predicting Plant Output Power")
st.markdown("Change the temperature,exhaust vaccum,ambient pressure,relative humidity to get the Electrical power in MW")

# inserting the  sliders for this 
st.header("CCPP Electrical Power Features")
col1, col2 = st.columns(2)
with col1:
    temp = st.slider('Temperature (C)',0,50,1)
    vaccum = st.slider('Exhaust Vaccum (cm Hg)',20,90,1)

with col2:
    pressure = st.slider('Ambient Pressure (milibar)',800,1200,20)
    humidity = st.slider('Relative Humidity (%)',10,140,10)

if st.button("Predict Electrical Power"):
    result = predict(np.array([temp,vaccum,pressure,humidity]))
    st.text(result[0])

# run:  streamlit run app.py

