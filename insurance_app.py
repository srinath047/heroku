# -*- coding: utf-8 -*-
"""
Created on Wed Jun 23 12:02:40 2021

@author: admin
"""
from pycaret.regression import load_model, predict_model
import streamlit as st
import pandas as pd
import numpy as np
import base64

model = load_model('insurance')

def predict(model, input_df):
    prediction_df = predict_model(estimator=model, data=input_df)
    predictions = prediction_df['Label'][0]
    return predictions

def run():
    
    from PIL import Image
    # image = Image.open('AI.png').convert('RGB').save('new.jpeg')
    image = Image.open('AI.png').convert('RGB')
    image_hospital = Image.open('hospital.png').convert('RGB')
    
    st.image(image,use_column_width=False)
    
    add_selectbox = st.sidebar.selectbox("How would you like to predict?", ("Online", "Batch"))
    
    st.sidebar.info('This app is created to predict patient hospital charges')
    st.sidebar.success('Good Day!')
    
    st.sidebar.image(image_hospital)
    
    st.title("Insurance Charges Prediction App")
    
    if add_selectbox == "Online":
        
        age = st.number_input('Age', min_value=1, max_value=100, value=25)
        sex = st.selectbox('Gender', ['male','female'])
        bmi = st.number_input('BMI', min_value=10,max_value=50, value=10)
        childern = st.selectbox('Children',[0,1,2,3,4,5,6,7,8,9,10])
        if st.checkbox('Smoker'):
            smoker = 'yes'
        else:
            smoker = 'no'
        region = st.selectbox('Region',['southwest','northwest','northeast','southeast'])
        
        output=""
        
        input_dict = {'age':age,'sex':sex, 'bmi':bmi, 'children':childern, 'smoker':smoker,'region':region}
        input_df = pd.DataFrame([input_dict])
        
        if st.button("Predict"):
            output = predict(model=model,input_df=input_df)
            output = "$"+str(output)
        
        st.success('The output is {}'.format(output))
        
        
    if add_selectbox == "Batch":
        
        file_upload = st.file_uploader("Upload csv file for predictions", type=["csv"])
        
        if file_upload is not None:
            data = pd.read_csv(file_upload)
            st.write(data)
        if st.button("Predict"):
            predictions = predict_model(estimator=model, data=data)
            st.write("Predictions are in the column Label")
            st.write(predictions)
            csv = predictions.to_csv(index=False)
            b64 = base64.b64encode(csv.encode()).decode()
            href = f'<a href="data:file/csv;base64,{b64}">Download CSV File</a> (right-click and save as &lt;some_name&gt;.csv)'
            st.markdown(href, unsafe_allow_html=True)
            

if __name__=='__main__':
    run()
    