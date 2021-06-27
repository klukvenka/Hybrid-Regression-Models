#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import numpy as np # линейная алгебра

import matplotlib.pyplot as plt # графики, анализ данных
import seaborn as sns # также для анализа
from scipy.stats import norm
from scipy import stats


# In[2]:


from sklearn.neighbors import KNeighborsRegressor 
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler # Scaling
import math
from sklearn.metrics import  r2_score
from sklearn.model_selection import GridSearchCV # Настройка гиперпараметров


# In[3]:


# Загружаем наши данные из csv файла
houses = pd.read_csv('data.csv') 


# In[4]:


houses.shape


# In[5]:


houses.dtypes


# In[6]:


houses.head()


# In[7]:


"Средняя цена дома ${:,.0f}".format(houses.price.mean())


# In[8]:


houses.corr()


# In[9]:


houses[houses==0].count()


# ## Замена домов с нулевой ценой

# In[10]:


houses[houses["price"]==0].head(50)


# In[11]:


# корреляция нулевой цены со всеми признаками
houses[houses["price"]==0].agg([min, max, 'mean', 'median'])


# In[12]:


# дома, признаки которых похожи на признаки домов, цена которых равна 0
houses1 = houses[(houses.bedrooms == 4) & (houses.bathrooms > 1) & (houses.bathrooms < 4) & (houses.sqft_living > 2500) & 
         (houses.sqft_living < 3000) & (houses.floors < 3) & (houses.yr_built < 1970)]


# In[13]:


houses1.shape


# In[14]:


houses1.price.mean()


# In[15]:


# Заменяем дома с нулевой ценой 
houses['price'].replace(to_replace = 0, value = 735000, inplace = True)
len(houses[(houses['price'] == 0)])


# ## Замена домов с 0 спальнями

# In[16]:


houses[houses["bedrooms"]==0]


# In[17]:


houses_0beds_bath = houses[(houses.price > 1090000) & (houses.price < 1300000) & (houses.sqft_living > 3000) & 
         (houses.sqft_living < 4900) & (houses.floors >= 2) &(houses.floors < 4) ]


# In[18]:


houses_0beds_bath.bedrooms.mean()


# In[19]:


# Заменяем дома с 0 количеством спален 
houses['bedrooms'].replace(to_replace = 0, value = 4, inplace = True)
len(houses[(houses['bedrooms'] == 0)])


# In[20]:


houses_0beds_bath.bathrooms.mean()


# In[21]:


# Заменяем дома с 0 количеством ванных комнат
houses['bathrooms'].replace(to_replace = 0, value = 3, inplace = True)
len(houses[(houses['bathrooms'] == 0)])


# ## Отношения между признаками. Выбросы

# In[22]:


ax = sns.pairplot(houses)


# In[23]:


ax = sns.distplot(houses['price'])
ax.set_title('Распределение цены', fontsize=14)


# In[24]:


# 
data = pd.read_csv('data.csv') 
houses['price'] = houses['price'].replace([data['price'][np.abs(stats.zscore(data['price'])) > 3]],np.median(houses['price']))


# In[25]:


ax = sns.distplot(houses['price'])
ax.set_title('Распределение цены', fontsize=14)


# In[26]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=houses, x="sqft_living", y="price")
ax.set_title('Площадь VS Цена', fontsize=14)


# In[27]:


houses['sqft_living'] = np.where((houses.sqft_living >6000 ), 6000, houses.sqft_living)


# In[28]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=houses, x="sqft_living", y="price")
ax.set_title('Площадь VS Цена', fontsize=14)


# In[29]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=houses, x="sqft_lot", y="price")
ax.ticklabel_format(style='plain')
ax.set_title('Площадь территории VS Цена', fontsize=14)


# In[30]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=houses, x="sqft_above", y="price")
ax.ticklabel_format(style='plain')
ax.set_title('Площадь крыши VS Цена', fontsize=14)


# In[31]:


houses['sqft_above'] = np.where((houses.sqft_above >5000 ), 5000, houses.sqft_above)


# In[32]:


plt.figure(figsize=(15,6))
ax = sns.scatterplot(data=houses, x="sqft_basement", y="price")
ax.ticklabel_format(style='plain')
ax.set_title('Площадь подвала VS Цена', fontsize=14)


# In[33]:


houses['sqft_basement'] = np.where((houses.sqft_basement >2000 ), 2000, houses.sqft_basement)


# ## Отношения между признаками. Коллинеарность

# In[34]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=houses['bedrooms'], y=houses['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Кол-во спален VS Цена', fontsize=14)


# In[35]:


# разбивка цен по каждой группе спален. len - кол-во домов; min,max - минимальная/максимальная цена
bedroom = houses.groupby(['bedrooms']).price.agg([len, min, max])
bedroom


# In[36]:


# сгруппировка домов с 7, 8, 9 спальнями с домами с 6 спальнями
houses['bedrooms'] = np.where((houses.bedrooms >6 ), 6, houses.bedrooms)


# In[37]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=houses['bathrooms'], y=houses['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Кол-во ванных комнат VS Цена', fontsize=14)


# In[38]:


houses[houses['bathrooms']==1.0].count()


# In[39]:


houses['bathrooms'] = np.where((houses.bathrooms == 0.75), 1, houses.bathrooms)
houses['bathrooms'] = np.where((houses.bathrooms == 1.25 ), 1, houses.bathrooms)
houses['bathrooms'] = np.where((houses.bathrooms > 4.75 ), 5, houses.bathrooms)


# In[40]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=houses['floors'], y=houses['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Этажи VS Цена', fontsize=14)


# In[41]:


floor = houses.groupby(['floors']).price.agg([len , min, max])
floor


# In[42]:


houses['floors'] = np.where((houses.floors == 3.5 ), 3, houses.floors)


# In[43]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=houses['waterfront'], y=houses['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Набережная VS Цена', fontsize=14)


# In[44]:


waterfront = houses.groupby(['waterfront']).price.agg([len , min, max])
waterfront


# In[45]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=houses['view'], y=houses['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Вид VS Цена', fontsize=14)


# In[46]:


view = houses.groupby(['view']).price.agg([len , min, max])
view


# In[47]:


plt.figure(figsize=(15,6))
ax = sns.barplot(x=houses['condition'], y=houses['price'])
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
ax.set_title('Состояние VS Цена', fontsize=14)


# In[48]:


condition = houses.groupby(['condition']).price.agg([len , min, max])
condition


# In[49]:


houses['condition'] = np.where((houses.condition == 1 ), 2, houses.condition)


# In[50]:


# вычеркиваем некоторые признаки из анализа
houses.drop(["date",'yr_built','yr_renovated','sqft_lot'], axis=1, inplace = True)


# In[51]:


plt.figure(figsize=(15,6))
ax = sns.heatmap(houses.corr(),annot = True)
ax.set_title('Корреляционная матрица', fontsize=14)


# In[52]:


houses.drop(['waterfront','condition','sqft_above'],axis=1, inplace=True)


# In[53]:


houses.dtypes


# In[54]:


houses.country.value_counts()


# In[55]:


houses['street'].nunique()


# In[56]:


plt.figure(figsize=(15,10))
ax = sns.barplot(x="city", y="price", data=houses)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right");


# In[57]:


plt.figure(figsize=(15,10))
ax = sns.barplot(x="statezip", y="price", data=houses)
ax.set_xticklabels(ax.get_xticklabels(), rotation=90, ha="right");


# In[58]:


houses.drop(['street','city','country'],axis=1, inplace=True)


# ## ZIPCODE encoding

# In[59]:


houses = pd.get_dummies(houses, columns=['statezip'], prefix = ['statezip'])

houses.head()


# In[60]:


houses.shape


# In[61]:


X1 = houses.drop(['price', 'bedrooms', 'bathrooms', 'sqft_living', 'floors', 'view', 'sqft_basement'],axis = 1)
y = houses["price"]


# In[62]:


import scipy.stats as stats
for i in X1.columns:
    print(stats.f_oneway(X1[i],y))


# # SCALING
# 

# In[63]:


houses['log_price'] = np.log(houses['price'])
houses = houses.drop(["price"],axis = 1)
houses.head()


# In[64]:


# WITH THE LOG OF THE PRICE -> THE RESULT IS BETTER
X = houses.drop(["log_price"],axis = 1)
y = houses["log_price"]


# In[67]:


X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)


# In[68]:


y_train


# Features scaling

# In[69]:


## Here the best is Standard
from sklearn.preprocessing import MinMaxScaler, StandardScaler, MaxAbsScaler, RobustScaler, QuantileTransformer
scaler = StandardScaler()

x_train_scaled = scaler.fit_transform(X_train)
X_train = pd.DataFrame(x_train_scaled)

x_test_scaled = scaler.fit_transform(X_test)
X_test = pd.DataFrame(x_test_scaled)


# ## KNN

# Choosin k

# In[71]:


# Define our candidate hyperparameters
hp_candidates = [{'n_neighbors': [2,3,4,5,6,7,8,9,10,11,12,13,14], 'weights': ['uniform','distance']}]

knn = KNeighborsRegressor()

# Search for best hyperparameters
grid = GridSearchCV(estimator=knn, param_grid=hp_candidates, cv=5, scoring='r2')
grid.fit(X_train, y_train)
# Get the results
print(grid.best_score_)
print(grid.best_estimator_)
print(grid.best_params_)


# Training

# In[78]:


knn = KNeighborsRegressor(n_neighbors=7, weights='distance')
# fit the model using the training data and training targets
knn.fit(X_train, y_train)


# In[79]:


knn.score(X_test, y_test)


# # SVR

# In[80]:


svr = SVR(kernel='rbf',  C=1e0, gamma=0.01)
svr.fit(X_train, y_train)


# In[83]:


grid_sv = GridSearchCV(svr, param_grid={"C": [1e0, 1e1, 1e2, 1e3], "gamma": np.logspace(-2, 2, 5)}, scoring='r2')
grid_sv.fit(X_train, y_train)
print("Best classifier :", grid_sv.best_estimator_)


# In[81]:


svr.score(X_test, y_test)


# In[84]:


print(grid_sv.best_score_)
print(grid_sv.best_estimator_)
print(grid_sv.best_params_)


# # HYBRID

# In[86]:


knn_pred = knn.predict(X_test)


# In[87]:


svr_pred = svr.predict(X_test)


# In[88]:


combine_pred = knn_pred*0.4 + svr_pred*0.6


# In[89]:


r2_score(y_test, combine_pred)


# In[ ]:




