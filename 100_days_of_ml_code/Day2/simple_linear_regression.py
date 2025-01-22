# In this regression mode, we are trying to minimize the errors in prediction by finding the "line of best fit" - the regression line from the errors would be minimal. 
import sys
sys.path.append('..')
# Import preprocessing module
from Day1.preprocessing import DataPreprocessor
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

processor = DataPreprocessor()

X, y = processor.load_data("studentscores.csv")
X_train, X_test, y_train, y_test = processor.split_data()

regressor = LinearRegression()
regressor = regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

fig, axes = plt.subplots(1, 2, figsize = (12, 6))


axes[0].scatter(X_train, y_train, color = 'red', label = 'Training Data')
axes[0].plot(X_train, regressor.predict(X_train), color = 'blue', label = 'Regression Line')
axes[0].set_title('Training Data')
axes[0].set_xlabel('X_train')
axes[0].set_ylabel('y_train')
axes[0].legend()


axes[1].scatter(X_test, y_test, color= 'red', label = 'Test Data')
axes[1].plot(X_test, y_pred, color = 'blue', label = 'Regression Line')
axes[1].set_title('Testing Data')
axes[1].set_xlabel('X_test')
axes[1].set_ylabel('y_test')
axes[1].legend()

plt.show()

