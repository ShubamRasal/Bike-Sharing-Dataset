#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# For displaying all the rows and columns

# In[2]:


pd.set_option('display.max.rows',500)
pd.set_option('display.max_columns',500)
pd.set_option('display.width',1000)
pd.set_option('display.expand_frame_repr',False)


# In[3]:


import os
os.chdir(r"D:\Data Analyst")
os.listdir()
bike_data = pd.read_csv("day - day.csv")
bike_data.head()


# In[4]:


bike_data.shape


# In[5]:


bike_data.info()


# In[6]:


bike_data.describe()


# In[7]:


bike_data.columns


# In[9]:


bike_data.rename(columns={'dteday':'date_day','yr':'year','mnth':'month','hum':'humidity','cnt':'count'},inplace=True)


# In[10]:


bike_data.head()


# # Exploratory Data Analysis

# In[11]:


import matplotlib.pyplot as plt
import seaborn as sns


# In[13]:


new_cols = ['season','year','month','holiday','weekday','workingday','weathersit','temp','atemp','humidity','windspeed','count']
bike_data_1=bike_data[new_cols]
bike_data_1.head()


# In[14]:


bike_data_1['season'] = bike_data_1['season'].map({1:'spring',2:'summer',3:'fall',4:'winter'})
plt.figure(figsize=[10,4])
sns.barplot(bike_data_1['season'],bike_data_1['count'])
plt.title('season_avg_cnt',fontsize=20)
plt.show()


# In[15]:


bike_data_1['year'] = bike_data_1['year'].map({0:'2018',1:'2019'})
sns.barplot(bike_data_1['year'],bike_data_1['count'])
plt.title('year_avg_cnt',fontsize=20)
plt.show()


# In[16]:


bike_data_1['month'] = bike_data_1['month'].map({1:'Jan',2:'Feb',3:'Mar',4:'April',
                                       5:'May',6:'June',7:'July',8:'Aug',
                                       9:'Sept',10:'Oct',11:'Nov',12:'Dec'})
plt.figure(figsize=[14,5])
sns.barplot(bike_data_1['month'],bike_data_1['count'])
plt.title('monthly_avg_cnt',fontsize=20)
plt.show()


# * Similar average count on august, june, september, july also on may, october repectively.
# * December, january, february have least demand because of winter season

# In[17]:


bike_data_1['weekday']=bike_data_1['weekday'].map({0:'Monday',1:'Tuesday',2:'Wednesday',3:'Thursday',4:'Friday',5:'Saturday',6:'Sunday'})
plt.figure(figsize=[14,5])
sns.barplot(bike_data_1['weekday'],bike_data_1['count'])
plt.title('avg_daily_cont')
plt.show()


# * Having higher demands on sunday, monday, saturday, friday 
# * People also prefer thruday, wednesday, tuesday

# In[18]:


bike_data_1['workingday']=bike_data_1['workingday'].map({0:"no_need",1:"need"})
sns.barplot(bike_data_1['workingday'],bike_data_1['count'])
plt.title('workingday_avg_cnt',fontsize=20)
plt.show()


# * Have similar demands

# In[19]:


bike_data_1['holiday']=bike_data_1['holiday'].map({0:'no_need',1:"need"})
plt.figure(figsize=[7,4])
sns.barplot(bike_data_1['holiday'],bike_data_1['count'])
plt.title('avg_cnt_depending_on_holiday',fontsize=20)
plt.show()


# In[20]:


bike_data_1['weathersit'] = bike_data_1['weathersit'].map({1:"Good/Clear",2:'Moderate/Misty',3:'Bad/LightRain',4:'Worse/HeavyRain'})
plt.figure(figsize=[12,5])
sns.barplot(bike_data_1['weathersit'],bike_data_1['count'])
plt.title('Avg_count_depending_on_weather', fontsize = 20)
plt.show()


# * When there is clear weather the demand is more

# In[21]:


sns.pairplot(bike_data_1,x_vars=['temp','atemp','humidity','windspeed'],y_vars='count',size=4,aspect=1,
            kind='scatter',diag_kind=None)
plt.show()


# Independent variables that are good predictor:
# * temp
# * weathersit
# * month
# * season
# * workingday

# In[22]:


bike_data_1.columns


# In[23]:


category = ['season','year','month','holiday','weekday','workingday','weathersit']

for i in category:
    bike_data_1[i] = bike_data_1[i].astype('category')


# In[24]:


bike_data_1.info()


# * Create dummies for linear model:

# In[25]:


dummy_bike_data = pd.get_dummies(bike_data_1[category],drop_first=True)
dummy_bike_data.head()


# In[26]:


bike_data_2 = pd.concat([bike_data_1,dummy_bike_data],axis=1)
bike_data_2.head()


# In[27]:


bike_data_2.drop(category,axis=1,inplace=True)


# In[28]:


bike_data_2.info()


# # Train - Test Split Data

# In[29]:


import sklearn
import statsmodels.api as sm
from sklearn.model_selection import train_test_split


# In[30]:


train_bike_data_2,test_bike_data_2 = train_test_split(bike_data_2, train_size=0.70, random_state=100)


# In[31]:


print(train_bike_data_2.shape)
print(test_bike_data_2.shape)


# # Rescaling the features

# In[32]:


from sklearn.preprocessing import MinMaxScaler


# In[33]:


scaler = MinMaxScaler()


# In[34]:


bike_data_2.columns


# In[36]:


numerical_variables = ['temp','atemp','humidity','windspeed','count']

train_bike_data_2[numerical_variables] = scaler.fit_transform(train_bike_data_2[numerical_variables])
train_bike_data_2[numerical_variables].head()


# In[37]:


train_bike_data_2.describe()


# In[38]:


plt.figure(figsize=[25,20])
sns.heatmap(train_bike_data_2.corr(),annot=True,cmap="Dark2")
plt.show()


# In[39]:


y_train = train_bike_data_2.pop("count")
X_train = train_bike_data_2

print(y_train.shape)
print(X_train.shape)


# In[41]:


from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression


# In[42]:


lm = LinearRegression()
lm.fit(X_train,y_train)

rfe = RFE(lm, 15)
rfe = rfe.fit(X_train, y_train)


# In[43]:


list(zip(X_train, rfe.support_, rfe.ranking_))


# In[44]:


rfe_col = X_train.columns[rfe.support_]
rfe_col


# In[45]:


X_train.columns[~rfe.support_]


# In[46]:


X_train_rfe = X_train[rfe_col]
X_train_rfe.head()


# # Model_No._1

# In[47]:


import statsmodels.api as sm
X_train_lm1 = sm.add_constant(X_train_rfe)

lr1 = sm.OLS(y_train, X_train_lm1).fit()

lr1.summary()


# In[50]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train_rfe.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe.values,i) for i in range(X_train_rfe.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Model_No._2

# * Removing feature humidity due to very high VIF values as all the p-values < 0.05

# In[52]:


X_train_rfe2 = X_train_rfe.drop('humidity', axis=1)

X_train_lm2 = sm.add_constant(X_train_rfe2)

lr2 = sm.OLS(y_train, X_train_lm2).fit()

lr2.summary()


# In[54]:


from statsmodels.stats.outliers_influence import variance_inflation_factor

vif = pd.DataFrame()
vif['Features'] = X_train_rfe2.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe2.values, i) for i in range(X_train_rfe2.shape[1])]
vif['VIF'] = round(vif['VIF'],2)
vif = vif.sort_values(by = 'VIF',ascending=False)
vif


# # Model_No._3

# * Removing Feature 'weathersit_Moderate/Mistry' due to very high VIF values as all the p-values < 0.05

# In[57]:


X_train_rfe3 = X_train_rfe2.drop('weathersit_Moderate/Misty', axis = 1)

X_train_lm3 = sm.add_constant(X_train_rfe3)

lr3 = sm.OLS(y_train, X_train_lm3).fit()

lr3.summary()


# In[58]:


vif = pd.DataFrame()
vif['Features'] = X_train_rfe3.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe3.values, i) for i in range(X_train_rfe3.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Model_No._4

# * Removing feature 'month_nov' because high p-value(0.148)

# In[59]:


X_train_rfe4 = X_train_rfe3.drop('month_Nov',axis=1)

X_train_lm4 = sm.add_constant(X_train_rfe4)

lr4 = sm.OLS(y_train, X_train_lm4).fit()

lr4.summary()


# In[60]:


vif = pd.DataFrame()
vif['Features'] = X_train_rfe4.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe4.values, i) for i in range(X_train_rfe4.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Model_No._5

# * Removing feature 'month_dec' because of high p-value(0.231)

# In[62]:


X_train_rfe5 = X_train_rfe4.drop('month_Dec', axis = 1)

X_train_lm5 = sm.add_constant(X_train_rfe5)

lr5 = sm.OLS(y_train, X_train_lm5).fit()

lr5.summary()


# In[63]:


vif = pd.DataFrame()
vif['Features'] = X_train_rfe5.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe5.values, i) for i in range(X_train_rfe5.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Model_No._6

#  * Removing feature 'month_Jan' because of high p-values(0.089)

# In[64]:


X_train_rfe6 = X_train_rfe5.drop('month_Jan', axis = 1)

X_train_lm6 = sm.add_constant(X_train_rfe6)

lr6 = sm.OLS(y_train, X_train_lm6).fit()

lr6.summary()


# In[65]:


vif = pd.DataFrame()
vif['Features'] = X_train_rfe6.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe6.values, i) for i in range(X_train_rfe6.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Model_No._7

# * Removing feature 'month_July' because of high collinearity with temp variable

# In[67]:


X_train_rfe7 = X_train_rfe6.drop('month_July', axis = 1)

X_train_lm7 = sm.add_constant(X_train_rfe7)

lr7 = sm.OLS(y_train, X_train_lm7).fit()

lr7.summary()


# In[68]:


vif = pd.DataFrame()
vif['Features'] = X_train_rfe7.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe7.values, i) for i in range(X_train_rfe7.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Model_No._8

# * Removing feature 'season_spring' because of high negative collinarity with temp variable

# In[71]:


X_train_rfe8 = X_train_rfe7.drop('season_spring', axis = 1)

X_train_lm8 = sm.add_constant(X_train_rfe8)

lr8 = sm.OLS(y_train, X_train_lm8).fit()

lr8.summary()


# In[72]:


vif = pd.DataFrame()
vif['Features'] = X_train_rfe8.columns
vif['VIF'] = [variance_inflation_factor(X_train_rfe8.values, i) for i in range(X_train_rfe8.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# # Validate Assumptions

# In[73]:


y_train_pred = lr8.predict(X_train_lm8)


# In[74]:


#residual calculations:
#--------------------------

res = y_train - y_train_pred

fig = plt.figure(figsize=[7,5])
sns.distplot((res),bins=20)
fig.suptitle('Error Terms',fontsize=20)
plt.xlabel('Errors',fontsize=18)
plt.show()


# * Therefore the residuals are normally distributed.
# * Our assumption for Linear Regression is valid

# # Check Homoscedasticity

# In[75]:


plt.figure(figsize=[8,5])
p = sns.scatterplot(y_train_pred,res)
plt.xlabel('y_pred/predicted values')
plt.ylabel('Residuals')

p = sns.lineplot([0,1],[0,0],color='red')
p = plt.title('Residuals VS fitted values plot for homoscedasticity check',fontsize=20)


# # Making Prediction Using Final Model

# In[79]:


test_bike_data_2[numerical_variables] = scaler.transform(test_bike_data_2[numerical_variables])
test_bike_data_2.head()


# In[80]:


test_bike_data_2.describe()


# In[81]:


y_test = test_bike_data_2.pop('count')
X_test = test_bike_data_2

print(y_test.shape)
print(X_test.shape)


# In[82]:


col_test = X_train_rfe8.columns

X_test = X_test[col_test]

X_test_lm8 = sm.add_constant(X_test)

X_test_lm8.head()


# In[83]:


y_test_pred = lr8.predict(X_test_lm8)


# # Model Evaluation

# In[85]:


fig = plt.figure()
plt.scatter(y_test, y_test_pred, alpha=.5)
fig.suptitle('y_test VS y_test_pred',fontsize=20)
plt.xlabel('y_test',fontsize=18)
plt.ylabel('y_test_pred',fontsize=16)


# * We do have linear relationship between y_test and y_test_pred

# # Residual Analysis

# In[86]:


from sklearn.metrics import r2_score

r2_test = r2_score(y_test, y_test_pred)
r2_train = r2_score(y_train, y_train_pred)

print('Test data r^2 :',round((r2_test*100),2))
print('Train data r^2 :',round((r2_train*100),2))


# # Adjusted R^2 Value for TEST

# In[87]:


# n for test data ,n1 for train data is number of rows
n = X_test.shape[0]
n1 = X_train_rfe8.shape[0]

# Number of features (predictors, p for test data, p1 for train data) is the number of columns
p = X_test.shape[1]
p1 = X_train_rfe8.shape[1]


# We find the Adjusted R-squared using the formula

adjusted_r2_test = 1-(1-r2_test)*(n-1)/(n-p-1)
adjusted_r2_train = 1-(1-r2_train)*(n1-1)/(n1-p1-1)

print('Test data adjusted r^2 :',round((adjusted_r2_test*100),2))
print('Train data adjusted r^2 :',round((adjusted_r2_train*100),2))


# # Final Result

# *Test data r^2 : 78.32
# 
# *Train data r^2 : 80.84
# 
# *Test data adjusted r^2 : 77.5
# 
# *Train data adjusted r^2 : 80.53

# # Below predictor variables influences bike booking:

# * Temp
# * September month
# * year 2019
# * Summer Season
# * Holiday
# * Weather is clear/Good
# * Wind_Speed

# # *** Thank You ***
