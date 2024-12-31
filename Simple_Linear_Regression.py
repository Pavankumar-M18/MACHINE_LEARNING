# import libraries

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import pickle

# importing dataset

dataset = pd.read_csv(r"C:\Users\pk161\OneDrive\DATA\Salary_Data.csv")
print(dataset)

# split the data into dependent and independent
# independent
x = dataset.iloc[:,:-1].values
print(x)

# dependent
y = dataset.iloc[:,-1].values
print(y)

# split the data into training set and testing set
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.20,random_state=0)


# creating the linear regression model
regressor = LinearRegression()
# fiting the model
regressor.fit(x_train,y_train)

# predict the model
y_pred = regressor.predict(x_test)

# visualize the training set
plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience(TRAINING SET)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# visualize test set
plt.scatter(x_test,y_test,color='red')
plt.plot(x_train,regressor.predict(x_train),color='blue')
plt.title("Salary vs Experience(TRAINING SET)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.show()

# compare actual vs prediction
comparision = pd.DataFrame({'Actual':y_test,'Predicted':y_pred})
print(comparision)

# predict salary for 12 and 20 years of experince
y_12 = regressor.predict([[12]])
y_20 = regressor.predict([[20]])

print(f"predicted Salary for 12 years of experience :${y_12[0]:,.2f}")
print(f"predicted salary for 20 years of experience :${y_20[0]:,.2f}")

# evaluating model performence
bias = regressor.score(x_train,y_train)
variance = regressor.score(x_test,y_test)

train_mse = mean_squared_error(y_train,regressor.predict(x_train))
test_mse = mean_squared_error(y_test,y_pred)

print(f"Training Score (R^2): {bias:.2f}")
print(f"Testing Score (R^2): {variance:.2f}")
print(f"Train MSE : {train_mse:.2f}")
print(f"Test MSE : {test_mse:.2f}")

# save the trained model
filename = "simple_linear_regression_model.pkl"
with open(filename,'wb') as file:
    pickle.dump(regressor,file)

print("Model has been pickled and saved as simple_linear_regression_model.pkl")
