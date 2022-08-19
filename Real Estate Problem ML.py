#!/usr/bin/env python
# coding: utf-8

# # REAL ESTATE PRICE PREDICTOR - ML
# 

# In[1]:


'''The data housing.data and housing.names have 13 features and 1 label (or predictor variable). The data has been saved 
in an excel file named - housing.data on the desktop.'''


# In[2]:


import pandas as pd


# In[3]:


housing = pd.read_csv("data.csv")


# ## ANALYSING / READING THE DATA FROM THE FILE

# In[4]:


housing.head()


# In[5]:


housing.info()

From the above information we can see that in all the columns(features) we have 506 datas each. So, no missing data
is reported
# In[6]:


housing['CHAS'].value_counts()


# In[7]:


housing.describe() 


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')

import matplotlib.pyplot as plt

'''The histogam shows number of observation for every value that are there in the features, 
in python you have to write (plt.show) to make the histogram visible to you'''

housing.hist(bins = 50, figsize=(20, 15))
# In[ ]:





# ## Training and Testing Data Splitting
import numpy as np
def split_train_test(data, test_ratio):
    np.random.seed(42)
    shuffled = np.random.permutation(len(data))
    print(shuffled)
    test_set_size = int( len(data) * test_ratio )
    test_indices = shuffled[:test_set_size]
    train_indices = shuffled[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices] 
'''data.iloc - means to return the data based on integer-location'''train_set, test_set = split_train_test(housing, 0.2)
print(f"Rows in training set : {len(train_set)}\nRows in test set :{len(test_set)}\n")
# # Now we will use the inbuilt scikit-learn library to do the above -but with fewer lines of codes. 

# In[9]:


from sklearn .model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size= 0.2, random_state= 42)


# the issue here - what if "CHAS" feature in Training set has recieved all the 0's and the no 1's. then the trained data is trained without 1's because we need train set to have all the possible values from the entire data set.
# This kind of specific shuffling can be achieved by - Stratified Shuffling Split

# In[10]:


from sklearn.model_selection import StratifiedShuffleSplit
split = StratifiedShuffleSplit(n_splits = 1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing['CHAS']):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]


# In[11]:


strat_test_set['CHAS'].value_counts()


# In[12]:


strat_train_set['CHAS'].value_counts()


# In[13]:


housing = strat_train_set.copy()


# ## FINDING CORRELATION

# In[14]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# Correlation above - it shows the correlation between MDEV(price of real estate - predictor) with each of the feature.
# 
# MEDV = 1 means a strong positive correlation. What you see above is the correlation in all the features from the given values in it's data set. The value of correlation will always lie between -1 to 1. 

# In[15]:


from pandas.plotting import scatter_matrix
attributes = ["MEDV","RM", "DIS", "LSTAT"]
scatter_matrix(housing[attributes], figsize=(12, 8))


# In[16]:


housing.plot(kind="scatter", x="RM", y ="MEDV")


# one of the benefits of drawing scatter plot for well correlating features is that, it helps us identify Outliers.

# In[17]:


housing['TAXRM'] = housing['TAX']/housing['RM']
housing.head()


# In[18]:


corr_matrix = housing.corr()
corr_matrix['MEDV'].sort_values(ascending=False)


# In[19]:


housing.plot(kind="scatter", x="TAXRM", y ="MEDV")


# In[20]:


housing = strat_train_set.drop("MEDV",axis = 1)
housing_labels = strat_train_set["MEDV"].copy()


# ## MISSING ATTRIBUTES

# Some of the RM datas are deleted before running this segment. To take care of missing attribute: 
# a) Get rid of missing attributes
# b) get rid of whole attribute
# c) set some sensible values - ideally (0, mean or median)

# a = housing.dropna(subset=['RM'])
# a.shape

# b = housing.drop("RM", axis = 1)

# In[21]:


#Option c
median = housing["RM"].median()
housing["RM"].fillna(median)


# In[22]:


housing.describe() #before we started imputing for the missing values


# In[23]:


'''Instead of doing the above for option - c, we simply use the inbuilt library of sklearn - imputer'''
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy ="median")
imputer.fit(housing)
imputer.statistics_
X = imputer.transform(housing)


# In[24]:


housing_tr = pd.DataFrame(X, columns = housing.columns)
housing_tr.describe() #Transformed Data Set of Housing


# # SCIKIT- LEARN DESIGNING

# Before this al we did was - we understood the features and the data.
# Primarily, three types of objects -
# 1. Estimators - estimates some parameters, based on imputer. eg. Imputer, it has fit method and transform method
# 1.1 Fit Method - fits the data set and computes the internal parameters.
# 
# 2. Transformers - transforms method takes input and returns output based on the learnings from fit(). It also has c convinience funtion called fit_transform(), which fits and then transforms. 
# 3. Predictors - eg. Linear Regression Model, fit() and predict() are common functions. It also gives some score function which will evaluate the predictions. 

# # Feature Scalling

# Two types of feature scaling methods - 
# 1. Min-Max Method (Normalization): (value-min)/(max-min) | this makes all the values to scale down from 0 to 1, but this method can get affected if there is a data error or too many outliers.
# calling function: MinMaxScaler from sklearn
# 
# 2. Standardization: (value - mean)/std | sklearn class called - StandardScaler, the changes in the values don't differ much in this method. 

# # Creating Pipeline
# pipelines are a structures of ML code which can help us to edit the code at a later stage with ease. It basically automates the entire model - without having to enter the data everytime when we want to do the analysis and predict. 

# In[25]:


from sklearn.pipeline import Pipeline
#let's scale the features for a maintained continuity amongst the dataset which is under analysis for ease of study.
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy = "median")),
    #........add as many as you want in your pipeline
    ('std_scaler', StandardScaler())
])


# In[26]:


housing_num_tr = my_pipeline.fit_transform(housing)
housing_num_tr #this is a numpy array
housing_num_tr.shape


# # Selecting a desired ML Model

# In[27]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor 
from sklearn.ensemble import RandomForestRegressor
#model = LinearRegression()
#model = DecisionTreeRegressor()
model = RandomForestRegressor()
model.fit(housing_num_tr, housing_labels)


# In[28]:


some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
prepared_data = my_pipeline.transform(some_data)
model.predict(prepared_data)


# In[29]:


some_labels


# # Evaluating the Model

# In[30]:


from sklearn.metrics import mean_squared_error
housing_predictions = model.predict(housing_num_tr)
mse = mean_squared_error(housing_labels, housing_predictions)
rmse = mse**(0.5)


# In[31]:


rmse


# 1. From above looking at the "Mean Squared Error" coming out to be '0'. We realise that our data is overfit in the model Decision Tree.
# 2. The MSE was equal to 4.589 in Linear Regression
# 3. Overfitting - it simply means that the model has also accomodated the noise and outliers in the data. A GOOD fit model does not learn the noice and the outliers but importantly UNDERSTANDS the trend in the data.
# 

# # Using Cross-Validation: Better Technique

# Cross-Validation: it is not a model, but it is a technique to better the fitting in the model itself, so it is used after the model has been defined.

# In[32]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, housing_num_tr, housing_labels, scoring= "neg_mean_squared_error", cv = 10)
rmse_scores = (-scores)**(0.5)


# In[33]:


def print_scores(scores):
    print("Scores are: ", scores)
    print("Mean: ", scores.mean())
    print("Standard Deviatioin: ", scores.std())


# In[34]:


print_scores(rmse_scores)


# # Saving the Model

# In[35]:


from joblib import dump, load
dump(model, 'MLProject.joblib')


# # Testing the model on - Real Estate Predictor Test Data

# In[36]:


x_test = strat_test_set.drop("MEDV", axis = 1 )
y_test = strat_test_set["MEDV"].copy()
x_test_prepared = my_pipeline.transform(x_test)
final_predictions = model.predict(x_test_prepared)
test_mse = mean_squared_error(y_test, final_predictions)
test_rmse = (test_mse)**(0.5)
test_rmse


# In[37]:


print("Finally Predicted Values:", final_predictions)


# In[38]:


print("Actual Values: ", list(y_test))

