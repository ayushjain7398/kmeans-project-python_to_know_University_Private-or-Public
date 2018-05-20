
# coding: utf-8

# 
# # K Means Clustering Project 
# **Cluster Universities into to two groups, Private and Public**
# 

# In[1]:


import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[2]:


df=pd.read_csv('College_Data',index_col=0)


# In[3]:


df.head()


# In[6]:


df.info()


# In[7]:


df.describe()


# ## EDA
# 

# In[13]:



sns.lmplot('Grad.Rate','Room.Board',data=df,hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)


# In[14]:



sns.lmplot('F.Undergrad','Outstate',data=df,hue='Private',palette='coolwarm',size=6,aspect=1,fit_reg=False)


# In[16]:


sns.set_style('darkgrid')
g = sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Outstate',bins=20,alpha=0.7)


# In[23]:


sns.set_style('darkgrid')
g=sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# In[24]:


df[df['Grad.Rate']>100]


# In[25]:


df['Grad.Rate']['Cazenovia College'] = 100


# In[26]:


df[df['Grad.Rate']>100]


# In[27]:


sns.set_style('darkgrid')
g=sns.FacetGrid(df,hue="Private",palette='coolwarm',size=6,aspect=2)
g = g.map(plt.hist,'Grad.Rate',bins=20,alpha=0.7)


# ## K Means Cluster Creation
# 
# 

# In[29]:


from sklearn.cluster import KMeans


# In[46]:



kmeans=KMeans(n_clusters=4)


# In[47]:


kmeans.fit(df.drop('Private',axis=1))


# In[48]:


kmeans.cluster_centers_


# ## Evaluation
# 
# 

# In[49]:


def con(clu):
    if clu=='Yes':
        return 1
    else:
        return 0


# In[50]:


df['Cluster']=df['Private'].apply(con)


# In[51]:


df.head()


# In[52]:


from sklearn.metrics import confusion_matrix,classification_report 
print(confusion_matrix(df['Cluster'],kmeans.labels_))
print(classification_report(df['Cluster'],kmeans.labels_))


# In[53]:


plt.scatter(df['Cluster'],df['Grad.Rate'],c=kmeans.labels_,cmap='rainbow')

