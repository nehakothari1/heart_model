#!/usr/bin/env python
# coding: utf-8

# In[62]:


import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
import warnings


# In[63]:


df=pd.read_csv(r"G:\Data Science Study\data sets\heart.csv")
df


# In[64]:




# In[ ]:


df.info()


# In[ ]:


df.describe()


# In[ ]:


df.isna()


# In[ ]:


df.isna().sum()


# In[ ]:


#here there are no nan or the missing values so no need to use mean,median,modeor knnimputer method


# In[ ]:


df['target'].value_counts()


# In[ ]:


plt.figure(figsize=(10,10))
sns.heatmap(df.corr(),annot=True)


# In[ ]:


x=df.drop('target',axis=1)
y=df['target']


# In[ ]:


x


# In[ ]:


y


# In[ ]:


x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42,stratify=y)


# In[ ]:


x_train


# In[ ]:


y_train


# In[65]:


log_regression=LogisticRegression()
log_regression.fit(x_train,y_train)


# In[68]:


y_pred_test=log_regression.predict(x_test)


# In[69]:


y_pred_test


# In[ ]:


# In[70]:


#Testing Data Evalation

y_pred = log_regression.predict(x_test)

cnf_matrix = confusion_matrix(y_test,y_pred)
print("Confusion Matrix: \n",cnf_matrix)
print("*"*80)

accuracy = accuracy_score(y_test,y_pred)
print("Accuarcy : ",accuracy)
print("*"*80)

clf_report = classification_report(y_test,y_pred)
print("Classification Report :\n",clf_report)


# In[71]:


#Training Data Evalation

y_pred_train = log_regression.predict(x_train)

cnf_matrix = confusion_matrix(y_train,y_pred_train)
print("Confusion Matrix: \n",cnf_matrix)
print("*"*80)

accuracy = accuracy_score(y_train,y_pred_train)
print("Accuarcy : ",accuracy)
print("*"*80)

clf_report = classification_report(y_train,y_pred_train)
print("Classification Report :\n",clf_report)


# In[73]:


with open('heart.pkl' ,'wb') as file:
    pickle.dump(log_regression,file)


# In[ ]:




