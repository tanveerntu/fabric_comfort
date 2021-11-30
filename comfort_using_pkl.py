# Before creating this .py file, the prediction models were built in google colab, using pycaret library
# the related file created in google colab for building models is saved as total_comfort.ipynb
# First step in model building is to build it in google colab using pycaret library
# and pickle out .pkl file of the model and save it in the same director as this .py file
# as a next step given in this .py file, the .pkl model is loaded and used for predictions based on user input
# this file will be run in anaconda command prompt using command: streamlit run comfort_using_pkl.py
# the same file can then be deployed on internet web using streamlit sharing

from pycaret.regression import load_model
from pycaret.regression import predict_model

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ---- HIDE STREAMLIT STYLE ----
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)


# to load the models from the same PC folder where this .py file is saved.
# It is assumed that the models with .pkl extension are already saved in the same folder
# the .pkl files comprising the models were created in google colab using pycaret library and then pickled out
# and saved
loaded_model = pickle.load(open('etr_comfort_model.pkl', 'rb'))

#etr stands for extra tree regressor

# Creating the Titles and Image
st.title("Fabric Comfort for Summer Fabrics")

# taking user input
warp_count = st.slider('Warp Count (Ne)', 30, 170, 60)
weft_count = st.slider('Weft Count (Ne)', 30, 170, 60)
epi = st.slider('Ends per inch', 80, 140, 100)
ppi = st.slider('Picks per inch', 60, 100, 80)

# store the inputs

features = [epi, ppi, warp_count, weft_count]

# convert user inputs into an array for the model

int_features = [int(x) for x in features]
final_features = [np.array(int_features)]


# Final prediction
prediction_etr = loaded_model.predict(final_features)
#st.success(f'Total Cloting Comfort: {round(prediction_etr[0], 2)}')

st.metric(label = "Predicted Fabric Comfort Score (out of 100):", value = int(prediction_etr))





