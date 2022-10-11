#!/usr/bin/env python
# coding: utf-8

# In[3]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[2]:


url = 'https://bit.ly/drinksbycountry'


# In[3]:


pd.read_table(url)


# In[9]:


pd.read_table(url, sep=',')


# In[10]:


#load data from other folder


# In[12]:


pd.read_csv(beers.csv)


# In[14]:


pd.read_csv('C:\Users\Hussain PALLITHOTTUN\Downloads\archive\beers.csv')


# In[6]:


pd.read_csv(r'C:\Users\Hussain PALLITHOTTUN\Downloads\archive\beers.csv')


# In[16]:


data = pd.read_csv(r'C:\Users\Hussain PALLITHOTTUN\Downloads\archive\beers.csv')


# In[17]:


data.head()


# In[9]:


data = pd.read_csv(r'C:\Users\Hussain PALLITHOTTUN\Downloads\archive\beers.csv',index_col='Unnamed: 0')


# In[10]:


data.head()


# In[22]:


data.shape


# In[24]:


data.info()


# In[25]:


data.isna().sum()


# In[26]:


# filling missing values


# In[28]:


freqgraph = data.select_dtypes(include=['float'])
freqgraph.hist(figsize=(20,15))
plt.show()


# In[29]:


data.columns


# In[30]:


data['abv']=data['abv'].fillna(data['abv'].median())


# In[31]:


data.isna().sum()


# In[33]:


for i in ['ibu','style']:
    data[i]=data[i].fillna(data[i].median())


# In[34]:


data.groupby()


# In[35]:


data.info()


# In[36]:


data.groupby('name')['ounces'].mean()


# In[40]:


data.ounces.nunique()


# In[38]:


data.drop('style',axis=1, inplace =True)


# In[39]:


data.head()


# In[42]:


plt.boxplot(data['brewery_id'])
plt.title('brewery id')


# In[43]:


plt.boxplot(data['ounces'])
plt.title('ounces')


# In[44]:


Q1 = np.percentile(data['ounces'],25,interpolation='midpoint')
Q2 = np.percentile(data['ounces'],50,interpolation='midpoint')
Q3 = np.percentile(data['ounces'],75,interpolation='midpoint')


# In[45]:


print(Q1)


# In[46]:


print(Q2)
print(Q3)


# In[48]:


data['ounces'].median()


# In[49]:


IQR = Q3-Q1


# In[50]:


low_lim = Q1-1.5*IQR
up_lim = Q3+1.5*IQR


# In[51]:


print(low_lim)
print(up_lim)


# In[52]:


outlier= []
for x in data['ounces']:
    if((x>up_lim) or (x<low_lim)):
        outlier.append(x)


# In[54]:


outlier


# In[57]:


ind1 = data['ounces']>up_lim
data.loc[ind1].index


# In[60]:


data.drop([477,  581,  957, 1181, 1341, 1342, 1343, 1344, 1345, 1346, 1347,
            1348, 1361, 1370, 1374, 1375, 1376, 1377, 1382, 1383, 1554, 1861,
            1895, 1896, 2097, 2099, 2100], inplace =True) 


# In[61]:


plt.boxplot(data['ounces'])
plt.title('ounces')


# In[63]:


data.shape


# In[64]:


#Encoding - One hot Encoding


# In[66]:


data.name.nunique()


# In[5]:


data = pd.get_dummies(data)


# In[68]:


data.head()


# In[14]:


corrmatrix = data.corr()
plt.subplots(figsize=(8,8))
sns.heatmap(corrmatrix, vmin=0.4, vmax=0.9, annot=True, linewidths=0.2, cmap='YlGnBu')


# In[15]:


data.columns


# In[38]:


y= data['ounces']
X=data.drop('ibu', axis=1)


# In[39]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, test_size=0.2)


# In[42]:


from sklearn import linear_model
lr = linear_model.LinearRegression()
model = lr.fit(X_train, y_train)
predictions = model.predict(X_test)


# In[46]:


data1 = pd.read_csv(r'https://raw.githubusercontent.com/justmarkham/pandas-videos/master/data/drinks.csv')


# In[47]:


#standard scaling


# In[52]:


data1.describe()


# In[53]:


type(data1
)


# In[54]:


from sklearn import preprocessing


# In[56]:


standardisation = preprocessing.StandardScaler()
data1 = standardisation.fit_transform(data1)


# In[57]:


# min max scaling


# In[58]:


x1 = data1.drop('total_litres_of_pure_alcohol', axis=1)


# In[59]:


x1.describe()


# In[61]:


min_max = preprocessing.MinMaxScaler(feature_range=(0,1))


# In[62]:


x= min_max.fit_transform(x1)


# In[63]:


#merging


# In[64]:


df1=pd.Dataframe{{"Sachin":[60,95,65,32,105),
"sanju kan11":47,45,12,64,45 36,44,96,9,11] ,
Index[2000,2010,2011,2012,2013 )
      
      


# In[ ]:


Jf2 pel.DuLaframe{{"Sachüs":( 02,95,65,32,105),
"Setwag":87,45,12,64,451 35,44,56,91,50),
"Dont [ ),
Index [2005,2006,2007,2008,2009 )

