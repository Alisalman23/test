import streamlit as st
import numpy as np
from sklearn.ensemble import RandomForestClassifier

st.title("Streamlit ML Model Deployment")

X = np.array([[-1, -1], [-2, -1], [-3, -2], [1, 1], [2, 1], [3, 2]])
Y = np.array([1, 1, 1, 2, 2, 2])

clf = RandomForestClassifier()
clf.fit(X, Y)

if st.checkbox("Show model parameters"):
    st.write(clf.get_params())

if st.button("Make Prediction"):
    x_input = st.text_input("Enter X value", "-1,-1")
    x_input = np.array(list(map(float, x_input.split(",")))).reshape(1, -1)
    prediction = clf.predict(x_input)
    st.write("Prediction:", prediction[0])
