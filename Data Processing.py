#!/usr/bin/env python
# coding: utf-8

# ## **Data Preprocessing**

# ##### Data preprocessing is a process of preparing the raw data and making it suitable for a machine learning model. It is the first and crucial step while creating a machine learning model.
# 1. Getting the dataset
# 2. Importing libraries
# 3. Importing datasets
# 4. Finding Missing Data
# 5. Encoding Categorical Data
# 6. Splitting dataset into training and test set
# 7. Feature scaling

# **> Getting the dataset**
# * To use the dataset in our code, we usually put it into a CSV file. 

# **> Import librery**
# * In order to perform data preprocessing using Python, we need to import some predefined Python libraries. These libraries are used to perform some specific jobs.

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# **> Importing the dataset**
# * Now we need to import the datasets which we have collected for our machine learning project. 

# In[7]:


dataset = pd.read_csv('Data_data_processing.csv') 
X = dataset.iloc[:,:-1].values #[rows,column] -> all rows and columns expect last column
Y = dataset.iloc[:,3].values #[rows,column] -> all rows and last columns 


# **> Resolving missing data**
# * The next step of data preprocessing is to handle missing data in the datasets. If our dataset contains some missing data, then it may create a huge problem for our machine learning model. Hence it is necessary to handle missing values present in the dataset.

# In[8]:


from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values=np.nan,strategy='mean') # missing places will be filled with mean value of the column
imputer = imputer.fit(X[:,1:3])
X[:,1:3] = imputer.transform(X[:,1:3]) 


# **> Encoding categorical data**
# * Since machine learning model completely works on mathematics and numbers, but if our dataset would have a categorical variable, then it may create trouble while building the model. So it is necessary to encode these categorical variables into numbers.

# In[14]:


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelEncoder_X = LabelEncoder()
X[:,0] = labelEncoder_X.fit_transform(X[:,0])
onehotencoder = ColumnTransformer([("Country", OneHotEncoder(), [0])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X)
labelEncoder_Y = LabelEncoder()
Y = labelEncoder_Y.fit_transform(Y)


# **> Split dataset in Training and Testing**
# * In machine learning data preprocessing, we divide our dataset into a training set and test set. This is one of the crucial steps of data preprocessing as by doing this, we can enhance the performance of our machine learning model.

# In[17]:


from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size=0.2, random_state=0) 


# **> Feature Scaling -> Putting variables in a static range to eliminate bais**
# * Feature scaling is the final step of data preprocessing in machine learning. It is a technique to standardize the independent variables of the dataset in a specific range. In feature scaling, we put our variables in the same range and in the same scale so that no any variable dominate the other variable.

# In[18]:


from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)


# #### In the above code, we have included all the data preprocessing steps together. But there are some steps or lines of code which are not necessary for all machine learning models. 
