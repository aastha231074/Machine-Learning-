# Step 1: Import Libraries 
# for mathematical functions 
import numpy as np 
# import and manage the data sets
import pandas as pd
# Imputer transformer for completing missing values 
from sklearn.impute import SimpleImputer 
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

#################################
# Step 2: Import your data set 
data = pd.read_csv("Data.csv")
X = data.iloc[:,:-1].values 
y = data.iloc[:,-1].values 

#################################
# Step 3: Handling the missing data 
# missing_value: parameter specifies which value in your dataset should be considered as "missing"
# strategy: mean, median, most_frequesnt, constant 
# fill_value: if startegy = constant then missing_values is replaced by the value in the fill_value (default = 0)
imputer = SimpleImputer(missing_values = np.nan, strategy = 'mean')
# purpose of fit method computes the imputation parameters (eg mean median mode etc)
imputer.fit(X[:,1:-1])
# the transform method applies the learned imputations 
X[:,1:-1] = imputer.transform(X[:,1:-1])

#################################
# Step 4: Encoding Categorical Variables 
label_encoder = LabelEncoder()
X[:,0] = label_encoder.fit_transform(X[:,0])

one_hot_encoder = OneHotEncoder(sparse_output = False)
X = one_hot_encoder.fit_transform(X[:, [0]])
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

#################################
# Step 5: Splitting the dataset into test set and training set 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 0)

#################################
# Step 5: Feature Scaling 
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)

