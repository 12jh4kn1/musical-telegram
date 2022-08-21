#!/usr/bin/env python
# coding: utf-8

# # CAR COST PREDICTOR - Dataset: CarDekho

# 1. CN - Car Name
# 2. YRS - years used
# 3. SP/PP - Selling Price and Present Price
# 4. KMSD - kilometers driven
# 5. FT - Fuel Type
# 6. ST - Seller Type
# 7. TRM - transmission
# 8. MIL - mileage (km/l)
# 9. ENG - engine 
# 10. MAXP - Maximum Power (bhp)
# 11. OWN - owner
# 12. SEAT - no of seats
# 13. The above are the names of the features of the cardekho dataset

# In[1]:


import pandas as pd
cardata = pd.read_csv("car data.csv")


# In[2]:


cardata.info()


# In[3]:


cardata['NOYRS'] = cardata['CY'] - cardata['YRS']
cardata.head()


# In[4]:


cardata["OWN"].value_counts()


# In[5]:


cardata["SEAT"].value_counts()

%matplotlib inline
cardata.hist(bins= 50, figsize = (20,15))
#This code is to plot the histogram chart to find the distribution of values over each feature
# In[6]:


corr_matrix = cardata.corr()
corr_matrix['PP'].sort_values(ascending = False)


# In[7]:


from pandas.plotting import scatter_matrix
attributes = ["PP", "SP","KMSD" ,"ENG","MAXP" ]
scatter_matrix(cardata[attributes], figsize = (12,8))


# # Splitting the Data into Training and Testing Set

# In[8]:


from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(cardata, test_size=0.2, random_state=42)


# In[9]:


cardata = train_set.drop("SP", axis = 1) #feature variable columns
cardata_labels = train_set["SP"].copy() #predictor variale column


# In[10]:


cardata.columns


# In[11]:


cardatafinal = cardata[[ 'PP', 'KMSD', 'OWN', 'MIL', 'ENG',
       'MAXP', 'SEAT', 'NOYRS']]
cardatafinal.head()


# In[12]:


test_set_final = test_set[['SP', 'PP', 'KMSD', 'OWN', 'MIL', 'ENG',
       'MAXP', 'SEAT', 'NOYRS']]
test_set_final.head()


# # Scaling the data and Creating Pipeline

# In[13]:


from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
my_pipeline = Pipeline([
       ('sta_scaler', StandardScaler())
])


# In[14]:


cardata_tr = my_pipeline.fit_transform(cardatafinal)


# # Selecting a desired ML Model

# In[52]:


from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
model = LinearRegression()
#model = DecisionTreeRegressor()
#model = RandomForestRegressor()
#model = GradientBoostingRegressor()
model.fit(cardatafinal, cardata_labels)


# In[53]:


from sklearn.metrics import mean_squared_error
model_predictions = model.predict(cardatafinal)
mse = mean_squared_error(cardata_labels, model_predictions)
rmse = mse**(0.5)
print(rmse)


# 1. Linear RMSE = 1.922616369616009
# 2. Decision Tree RMSE = 3.103167691559091e-18
# 3. Random Forest Regressor RMSE = 0.7550093217139772
# 4. Gradient Boosting Regressor RMSE = 0.3119487598128941
# 5. AdaBoost Tegressor RMSE = 0.9040293240082928

# A realistic least RMSE and suitable fit would be Gradient Boosting Regressor.

# # Using Cross Validation

# In[54]:


from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, cardatafinal, cardata_labels, scoring="neg_mean_squared_error", cv = 10)
rmse_scores = (-scores)**(0.5)
def print_scores(scores):
    print("Scores are: ", scores)
    print("Means are:" , scores.mean())
    print("Standard Deviation are:", scores.std())
    
#Calling the print scores of cross validated result functions

print_scores(rmse_scores)


# # Saving the Model

# In[55]:


from joblib import dump, load
dump (model, 'Carcost_Predictor.joblib')


# # Testing Model on a Test Set

# In[56]:


x_test = test_set_final.drop("SP", axis = 1 )
y_test = test_set_final["SP"].copy()
x_test_prepared = my_pipeline.transform(x_test) #imputing and standardizing the values
final_predictions = model.predict(x_test_prepared)
test_mse = mean_squared_error(y_test, final_predictions)
test_rmse = (test_mse)**(0.5)
print("The Tested Data RMSE is: ",test_rmse)


# In[57]:


print("Finally Predicted Values:", final_predictions)


# In[58]:


print("Actual Values: ", list(y_test))


# In[ ]:




