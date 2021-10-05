import pandas as pd
import numpy as np
import streamlit as st
import joblib
from PIL import Image as img
import os
from sklearn.tree import DecisionTreeClassifier


pwd = os.getcwd()
img_path = os.path.join(pwd,'static/creditcardDesign.png')
hero_img = img.open(img_path)
st.image(hero_img, use_column_width=True)
st.write("""
# CREDIT CARD FRAUD DETECTION 

 This Machine Learning Model Predicts Credit Card Transactions as ***FRAUDULENT or NON-FRAUDULENT*** 
""")
st.markdown("""[  Credit Card Fraud Training Dataset ] (https://drive.google.com/file/d/1GSCnWvDOtsJ8hE_b4KUXPmN-kbnlmZQT/view?usp=sharing)""")

st.write('#### Prediction Model Accuracy Score : 99.94% ')

st.header('User Input Variables')
st.markdown("""[Get Example Card Data Here](https://drive.google.com/file/d/1zeIlA9wIOhLbeaAOyMMRMdwSqQg8RFUQ/view?usp=sharing)""")


uploaded_file = st.file_uploader('upload CardData.csv file Here', type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file, sep=',')
    st.subheader("""
         Your Defined Features
        """)
    df = pd.DataFrame(input_df.values, columns=input_df.columns, index=[0])
    # print(df)
    st.write(df)


    model_filename = os.path.join(pwd, 'trained_model/trained_creditcard_fraud_detection.sav')
    model = joblib.load(filename=model_filename)
    result = model.predict(df)


    st.subheader('Predicted Result')
    result_map = {'Non-Fraudulent': 0, 'Fraudulent': 1}
    st.write(type(result))
    st.write(result)

    st.subheader('Prediction Class Type')
    if result == 0:
        st.write('## NON-FRAUDULENT')
    else:
        st.write('## FRAUDULENT')

else:
    st.write('still awaiting input file.........')


