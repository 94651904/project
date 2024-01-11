#!/usr/bin/env python
# coding: utf-8

# # Project Name:- Online Payments Fraud Detection Using Machine Learning

# # Batch:- ML10

# In[2]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


# In[3]:


df=pd.read_csv('PS_20174392719_1491204439457_loggggg.csv')
df


# In[8]:


df.head()


# In[9]:


df.tail()


# In[10]:


df.info()


# In[11]:


df.describe()


# In[12]:


df.isnull().sum()


# In[13]:


df.shape


# In[14]:


df.type.value_counts()


# In[15]:


type=df['type'].value_counts()


# In[16]:


transactions=type.index


# In[17]:


quantity=type.values


# In[18]:


type.plot(kind='bar')


# In[19]:


import plotly.express as px
px.pie(df,values=quantity,names=transactions,hole=0.4,title="Distribution of Transaction Type")


# In[20]:


df


# In[21]:


df['isFraud']=df['isFraud'].map({0:'No Fraud',1:'Fraud'})


# In[22]:


df


# In[23]:


df['type'].unique()


# In[24]:


df['type']=df['type'].map({'PAYMENT':1, 'TRANSFER':4, 'CASH_OUT':2, 'DEBIT':5, 'CASH_IN':3})


# In[25]:


df


# In[26]:


df['type'].unique()


# In[27]:


df.type.value_counts()


# In[24]:


x=df[['type','amount','oldbalanceOrg','newbalanceOrig']]
x


# In[28]:


y=df.iloc[:,-2]


# In[29]:


y # isfraud is my label


# In[30]:


X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.35,random_state=42)


# In[31]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()
model.fit(X_train,y_train)


# In[32]:


model.score(X_test,y_test)


# In[33]:


model.predict([[4,180,181,10]])


# In[34]:


data1 = np.array([[6, 80000, 80000, 0.0]])
model.predict(data1)


# In[35]:


from sklearn.naive_bayes import GaussianNB

model_gnb = GaussianNB()
model_gnb.fit(X_train,y_train)


# In[36]:


model_gnb.score(X_test,y_test)


# In[37]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train, y_train)


# In[38]:


model.score(X_test,y_test)


# In[39]:


from sklearn.model_selection import cross_val_score


# In[40]:


score_lr=cross_val_score(LogisticRegression(solver='liblinear',multi_class='ovr'), x,y,cv=3)
print(score_lr)
print("Avg :",np.average(score_lr))


# In[41]:


#x=df[['type','amount','oldbalanceOrg','newbalanceOrig']]
data = np.array([[5, 7880, 7880, 0.0]])
model.predict(data)


# In[42]:


data2 = np.array([[4,181,181,0.0]])
model.predict(data2)


# In[ ]:





# In[ ]:




