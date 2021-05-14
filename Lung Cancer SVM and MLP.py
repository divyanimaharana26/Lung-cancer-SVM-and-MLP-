#!/usr/bin/env python
# coding: utf-8

# # TOPIC : LUNG CANCER DETECTION USING SVM AND MLP
# AUTHOR: Divyani Maharana 

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
data=pd.read_csv("cancer patient data sets.csv" , sep=",")
data


# In[3]:


df=data.drop(['Patient Id','Age','Gender',	'Air Pollution','Dust Allergy','OccuPational Hazards','Balanced Diet','Obesity'],axis=1)


# In[4]:


df1=df.drop(['Fatigue','Weight Loss','Weight Loss',	'Shortness of Breath',	'Wheezing',	'Swallowing Difficulty','Clubbing of Finger Nails','Frequent Cold','Snoring','Unnamed: 26','Unnamed: 27','Smoking','Dry Cough','Genetic Risk','chronic Lung Disease','Passive Smoker','Chest Pain'],axis=1)


# In[5]:


df1


# In[6]:


df1.info()


# In[7]:


df1.describe()


# In[8]:


df1.isnull().sum()


# In[9]:


plt.figure(figsize=(20,20))
sns.heatmap(df1.isnull())


# In[10]:


corr=df1.corr()
corr['Result'].sort_values(ascending=False)


# In[11]:


# visualize correlation barplot
plt.figure(figsize = (16,15))
ax = sns.barplot(df1.corrwith(df1.Result).index, df1.corrwith(df1.Result))
ax.tick_params(labelrotation = 90)


# In[12]:


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
df1


# In[13]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1['Level']=le.fit_transform(df1['Level'])
df1


# In[14]:


X=df1.drop(['Result'],axis=1)
y=df1['Result']


# In[15]:


from sklearn.metrics import confusion_matrix, classification_report, accuracy_score


# In[16]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state= 45)


# In[17]:


# Feature scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)


# In[18]:


from sklearn.svm import SVC
svc_classifier = SVC(kernel='linear',C=0.25,random_state= 70)
svc_classifier.fit(X_train, y_train)
y_pred_scv = svc_classifier.predict(X_test)
accuracy_score(y_test, y_pred_scv)*100


# In[20]:


cm=confusion_matrix(y_test,y_pred_scv)
cm


# In[21]:


from sklearn.neural_network import MLPClassifier
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state= 45)


# In[22]:


clf=MLPClassifier(hidden_layer_sizes=(800,800),random_state=56)
clf.fit(X_train,y_train)
prediction=clf.predict(X_test)
report=classification_report(y_test,prediction)
print (report)
accuracy=clf.score(X_test,y_test)
print(accuracy*100)


# In[ ]:




