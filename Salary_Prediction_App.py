import streamlit as st
import pickle
import numpy as np

# load the saved model
model = pickle.load(open(r"C:\Users\pk161\VS_CODE\PYTHON_PROJECTS\MACHINE_LEARNING\simple_linear_regression_model.pkl",'rb'))

# set the title of the streamlit app
st.title("Salary Prediction App")

# add a brief description
st.write("This app predicts the salary based on years of experience using a simple linear regression model.")

# add input widget for users to enter years of experience
years_experience = st.number_input("Enter Years of Experience:", min_value=0.0,value=1.0,step=0.5)

# when the button is clicked, make prediction
if st.button("Predict Salary"):

    # make a prediction using the model
    experience_input = np.array([[years_experience ]])
    prediction = model.predict(experience_input)

    # display the result
    st.success(f"The predicted salary for{years_experience} years of experience is : ${prediction[0]:,.2f}")

st.write("The model was trained using a dataset of salaries and years of experience.")
    

