# kNN-real-estate-dataset


#import libraries
import pandas as pd 
import numpy as np 

#load the real estate data
df = pd.read_csv('real_estate_price_size_year_view.csv')

df.head()

#check number of rows and columns in dataset
df.shape

#create a dataframe with all training data except the target column
X = df.drop(columns=['view','year'])

#check that the target variable has been removed
X.head()

#separate target values
y = df['view']

#view target values
y.head()

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)

knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)

#show model predictions on the test data
knn.predict(X_test)

#check accuracy of model on the test data
knn.score(X_test, y_test)
