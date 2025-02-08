import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('Social_Network_Ads.csv')
X = data.iloc[:,1:-1].values
y = data.iloc[:,-1].values 

# Pre-processing 
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

ct = ColumnTransformer(
    transformers=[
        ('encoder', OneHotEncoder(drop = 'first'),[0])
    ],
    remainder='passthrough'
)

X = np.array(ct.fit_transform(X))

from sklearn.model_selection import train_test_split
X_train, X_test, y_tain, y_test = train_test_split(X,y,test_size=0.25, random_state=0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state=0)
classifier.fit(X_train, y_tain)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)

print(accuracy_score(y_test, y_pred))