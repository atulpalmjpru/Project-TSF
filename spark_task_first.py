#!/usr/bin/env python
# coding: utf-8

# # This is Task on linear Regression by TSF 
# 

# # Task 2 @TSF by Atul Pal

# In[1]:


#importing  library 
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


#reading data from file
base_url="http://bit.ly/w-data"
fhand=pd.read_csv(base_url)
print("Data imported Succesfully")
fhand.head(10)


# In[3]:


fhand.plot(x="Hours",y="Scores",style='d')
plt.title("Hours vs percentage")
plt.xlabel("Hours Studied")
plt.ylabel("Percentage Score")
plt.show()


# In[12]:


x=fhand.iloc[:,:-1].values
y=fhand.iloc[:,1].values
print(x)
print(y)


# In[5]:


# splitting data into train and test format
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)


# In[6]:


#Training the data for linear regression
from sklearn.linear_model import LinearRegression
myregressor=LinearRegression()
myregressor.fit(x_train,y_train)
print("Done")


# In[7]:


line=myregressor.coef_*x+myregressor.intercept_
plt.scatter(x,y)
plt.plot(x,line);
plt.show()


# In[8]:


print(x_test)
y_pred=myregressor.predict(x_test)


# In[9]:


#this is my own data set which I want to examine.
Hours=[[9.25],
       [6.55],
       [7.85],
       [8.79],
       [9.25]]
       
       
own_pred=myregressor.predict(Hours)
print("No of Hours = {}".format(Hours))
print("Predicted Score = {}".format(own_pred[:]))


# In[10]:


mydata=pd.DataFrame({'Actual':Hours,'Predicted':own_pred})
mydata


# In[11]:


from sklearn import metrics
print("Mean Absolute Error",metrics.mean_absolute_error(y_test,y_pred))


# # First Tech task done.....@Atul Pal with TSF

# In[ ]:




