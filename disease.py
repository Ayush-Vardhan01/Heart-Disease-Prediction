import numpy as np 
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# load the dataset
heart_data = pd.read_csv("heart.csv")

# number of rows and columns in the dataset
heart_data.shape

# checking for missing values
heart_data.isnull().sum()

# statistical measures about the data
heart_data.describe()

# checking the distribution of target variable
heart_data ['target'].value_counts()
# 1--> defective heart      ,   0--> healthy heart


# spilliting the features and target
X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']


# splitting the data into training data and test data
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

# MODEL TRAINING
model = LogisticRegression()
# logistic regression
model.fit(X_train,Y_train)


# Model Evaluation

# Accuracy Score
# accuracy on training data

X_train_prediction = model.predict(X_train)
training_data_prediction = accuracy_score(X_train_prediction,Y_train)


# accuracy on test data

X_test_prediction = model.predict(X_test)
test_data_prediction = accuracy_score(X_test_prediction,Y_test)


# build a predictive system
input_data = (57,1,0,130,131,0,1,115,1,1.2,1,1,3)

# change the input data to a numpy array
input_data_as_numpy_array = np.asarray(input_data)

# reshape the numpy array as we are predicting for only on instance
input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

prediction = model.predict(input_data_reshaped)


if (prediction[0] == 0):
    print("the person doesnt have a heart disease")

else:
    print("the person have a heart disease")