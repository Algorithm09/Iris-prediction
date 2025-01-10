# Implementing Machine learning model
import streamlit as st
import pickle
import pandas as pd
import os
import sklearn

file_path = os.path.join(os.path.dirname(__file__), "iris_prediction.pkl")
with open("C:/Users/ALGORITHM/main.py/MLearning/iris_prediction.pkl","rb") as file:
    model = pickle.load(file)

st.title("   Iris Flower prediction App")
st.header("This app predict the species of iris flower")
st.sidebar.header("User Input Parameter")
st.sidebar.info("choose the parameters for your predictions")
st.sidebar.write("This parameters help the model to predict the species of iris")

def user_input():
    sepal_length = st.sidebar.slider("sepal_length",4.3,7.9,5.4)

    sepal_width = st.sidebar.slider("sepal_length", 2.0,4.4,3.4)

    petal_length = st.sidebar.slider("petal_length",1.0, 6.9,1.3)

    petal_width = st.sidebar.slider("petal_width", 0.5,2.0,0.2)

    data = {"sepal_length":sepal_length, "sepal_width":sepal_width,
            "petal_length":petal_length, "petal_width":petal_width }
    features = pd.DataFrame(data,index=[0])
    return features

df = user_input()

st.markdown("### User input_parameters")

st.write(df)

st.subheader("Class label and their corresponding prediction values")

st.info("""Labels ----- prediction_values\n
Setosa ------- 0\n
Versicolor ------ 1\n
Virginica ------  2  """)
#setosa":0, 'versicolor':1 , 'virginica':2
prediction = model.predict(df)
pred = model.predict_proba(df)

if prediction == 0:
    st.subheader("Prediction")
    st.write(prediction)
    st.write("This is a Setosa flower")
    
    st.write("the probabilt of being setosa is ", pred)
    
    #st.write("The ro")

elif prediction == 1:
    st.subheader("prediction")
    st.write(prediction)
    st.write("This is a Veriscolor flower")
    st.write("The probabitlity of being veriscolor is", pred)

else:
    st.subheader("prediction")
    st.write(prediction)
    st.write("This is a Virginica flower")
    st.write("The probabitlity of being virginica is", pred)
