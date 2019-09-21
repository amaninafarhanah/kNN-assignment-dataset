# kNN-real-estate-dataset

#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import other libraries
import pandas as pd 
import numpy as np 


# In[2]:


# Load the country clusters data
df = pd.read_csv('real_estate_price_size_year_view.csv')


# In[3]:


df.head()


# In[4]:


#check number of rows and columns in dataset
df.shape


# In[52]:


#create a dataframe with all training data except the target column
X = df.drop(columns=['view','year'])

#check that the target variable has been removed
X.head()


# In[53]:


#separate target values
y = df['view']

#view target values
y.head()


# In[54]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


# In[63]:


#split dataset into train and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1, stratify=y)


# In[64]:


knn = KNeighborsClassifier(n_neighbors = 5)

knn.fit(X_train, y_train)


# In[65]:


#show first 5 model predictions on the test data
knn.predict(X_test)


# In[66]:


#check accuracy of our model on the test data
knn.score(X_test, y_test)


# In[ ]:




