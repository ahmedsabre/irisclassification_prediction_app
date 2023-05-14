import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
st.title('iris flower types prediction app')
st.sidebar.header('input parameters')
def input_features():
    sepal_length=st.sidebar.slider('Sepal Length',4.3,7.9,5.4)
    sepal_width=st.sidebar.slider('Sepal Width',2.0,4.4,3.4)
    petal_length=st.sidebar.slider('Petal Length',1.0,6.9,1.3)
    petal_width=st.sidebar.slider('Petal Width',0.1,2.5,0.2)
    data={'sepal_length':sepal_length,'sepal_width':sepal_width,'petal_length':petal_length,'petal_width':petal_width}
    features=pd.DataFrame(data,index=[0])
    return features
df=input_features()
st.subheader('input parameters')
st.write(df)
iris=datasets.load_iris()
x=iris.data
y=iris.target
clf=RandomForestClassifier()
clf.fit(x,y)
prediction=clf.predict(df)
prediction_proba=clf.predict_proba(df)
st.subheader('class names')
st.write(iris.target_names)
st.subheader('prediction')
if st.button('flower type'):
  st.write(iris.target_names[prediction],prediction_proba)
  






