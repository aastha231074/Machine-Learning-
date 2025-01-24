# Assumptions: 
# For a successful regression analyis it's seeential to validate these assumptions:
# 1. Linearity: The relationship between the dependent and independent variables should be linear 
# 2. Homoscedasticity: (constant  variance) of the errors should be maintained 
# 3. Multivarient Normality: Multiple regression assumes that the residuals are normally distributed 
# 4. Lack of Multicollinearity: It is assumed that there is little or no multicollinearity in the data. Multicollinearity occurs when the independent variables (features) are not not independent of each other 
import sys
sys.path.append('..')
# Import preprocessing module
import matplotlib.pyplot as plt
from Day1.preprocessing import DataPreprocessor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


processor = DataPreprocessor()

X, y = processor.load_data("50_Startups.csv")

ct = ColumnTransformer(transformers=[('encoder',OneHotEncoder(),[-1])], remainder='passthrough')
X = ct.fit_transform(X)

# Avoid Dummy Variable Trap
# So when you do one hot encoding you create dummy variales 
#  data = [['New York'], ['Los Angeles'], ['Chicago'], ['New York']]
# New York   Los Angles   Chicago
#    1           0           0
#    0           1           0
#    0           0           1
#    1           0           0
# If we keep all the three then the assumption of multicolinearity is voilated as one of the columns can be used to predict the other
# ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(drop='first'), [0])], remainder='passthrough')
X = X[: , 1:]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state = 1)
regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

print(y_pred)