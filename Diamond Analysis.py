#!/usr/bin/env python
# coding: utf-8

# Imports: 
# following imports and additional installations are needed: 

# In[50]:


import pandas as pd
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")
from pandas.plotting import scatter_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from random import randint
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score 
from sklearn.preprocessing import StandardScaler
from math import exp
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression


get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


df= pd.read_csv("diamonds_FSAI_SoSe22.csv")
df.head()


# In[3]:


df.info()


# In[4]:


df.shape


# In[5]:


df.drop(columns = 'Unnamed: 0', axis = 1, inplace = True )
df.head()


# In[6]:


df.shape


# In[7]:


df.info


# In[8]:


df["price"] = df["price"].astype(float)
df.head()


# In[9]:


df.describe()


# In[10]:


# x y z are giving us the value zero which we don't really need thats why we remove the rows which they are 0 

df = df.drop(df[df["x"]==0].index)
df = df.drop(df[df["y"]==0].index)
df = df.drop(df[df["z"]==0].index)


# In[11]:


df.shape


# In[12]:


df["cut"].value_counts()
# cut category 


# In[13]:


df['color'].value_counts()


# In[14]:


df['clarity'].value_counts()


# In[15]:


df.describe()
#a preview of the summary of the numerical attributes and then an histogram on the dataset.


# In[16]:


df.hist(bins = 50, figsize = (20, 15))
plt.show()


# In[17]:


sns.pairplot(df , diag_kind = 'kde');


# In[18]:


plt.figure(figsize = (10,5))
sns.heatmap(df.corr(),annot = True , cmap = 'Pastel2' );
plt.show()

# carat and price have the strongest correlation 
# after carat x,y,z have the strongest correlation 
# depth and table have the weakest correlations
# will use carat for Stratified Sampling.


# In[19]:


#test set

corr_matrix = df.corr()
plt.figure(figsize = (10,5))
corr_matrix['price'].sort_values(ascending = False).plot(kind = 'bar');


# In[20]:


#taking a closer look at the carat histogram 

df["carat"].hist(bins = 50)
plt.show()


# In[21]:


#diamonds are roughly between 0.3 and 1.5 Carats based on this we can divide them in 5 categories

df["carat_cat"] = np.ceil(df["carat"] / 0.35)

#merging categories 
df["carat_cat"].where(df["carat_cat"] < 5, 5.0, inplace = True)


# In[22]:


df["carat_cat"].value_counts()


# In[23]:


df["carat_cat"].hist()
plt.show()


# In[ ]:





# In[24]:


sns.pairplot(df[["price", "carat", "cut"]], hue = "cut", height = 5)
plt.show()
sns.barplot(x = "carat", y = "cut", data = df)
plt.show()
sns.barplot(x = "price", y = "cut", data = df)
plt.show()

# premium most expensive but less heavy 
# fair are the heaviest but second most expensive 
# ideal cheapest and the light in compar to other cuts! 


# In[ ]:





# In[25]:


sns.pairplot(df[["price", "carat", "clarity"]], hue = "clarity", height = 5)
plt.show()
sns.barplot(x = "carat", y = "clarity", data = df)
plt.show()
sns.barplot(x = "price", y = "clarity", data = df)
plt.show()


# T1 doesn't hold the highest  even though it is the most priced.
# Apart from I1 if the rest stays, the price of a diamond could fairly be relative to its clarity, to some extent.


# In[26]:


sns.pairplot(df[["price", "carat", "color"]], hue = "color", height = 5)
plt.show()
sns.barplot(x = "carat", y = "color", data = df)
plt.show()
sns.barplot(x = "price", y = "color", data = df)
plt.show()

# the color J is the most expensive one
# based on the 2 charts we can see that color of the diamond is very dependable to to price! 


# In[28]:


attributes = ["x", "y", "z", "table", "depth", "price"]
scatter_matrix(df[attributes], figsize=(25, 20))
plt.show()

# low correlation between depth and table with the price! 


# In[29]:


dia=df.copy()
dia


# In[30]:


diamonds = dia.drop("price", axis = 1)

# Set a new dataset label variable
diamond_labels = dia["price"].copy()

# Drop all the category, so we could have only numeric
diamonds_num = diamonds.drop(["cut", "color", "clarity"], axis = 1)
diamonds_num.head()


# In[31]:


# Perform the feature scaling on the numeric attributes of the dataset
num_scaler = StandardScaler()
diamonds_num_scaled = num_scaler.fit_transform(diamonds_num)

# Preview 
pd.DataFrame(diamonds_num_scaled).head()

#basically how our data will look during the process


# In[32]:


diamonds_cat = diamonds[["cut", "color", "clarity"]]
diamonds_cat.head()

# we only the category attributes to work with here


# In[33]:


# Perform the one-hot encoding on the category attributes of the dataset
cat_encoder = OneHotEncoder()
diamonds_cat_encoded = cat_encoder.fit_transform(diamonds_cat)

# Convert the encoded categories to arrays and Preview
pd.DataFrame(diamonds_cat_encoded.toarray()).head()


# In[39]:


#we do now to to merge the numeric feature scaled attributes and the encoded category attributes

num_attribs = list(diamonds_num)
cat_attribs = ["cut", "color", "clarity"]

# Pipeline to transform our dataset
pipeline = ColumnTransformer([
    ("num", StandardScaler(), num_attribs), # Perform feaured scaling on numeric attributes
    ("cat", OneHotEncoder(), cat_attribs) # Perform One-Hot encoding on the category attributes
])


# In[38]:


# Transformed dataset to feed the ML Algorithm
diamonds_ready = pipeline.fit_transform(diamonds)

# Preview
pd.DataFrame(diamonds_ready).head()


# In[52]:


models_rmse = [] # Holds Models original RMSE
cvs_rmse_mean = [] # Holds the Cross Validation RMSE Mean
tests_rmse = [] # Holds the tests RMSE
tests_accuracy = [] # Holds the tests accuracy
models = []

# Remove label from test set
X_test = diamonds.drop("price", axis = 1)
# Have label stand alone
y_test = diamonds["price"].copy()

def display_model_performance(model_name, model, diamonds = diamonds_ready, labels = diamond_labels,
                              models_rmse = models_rmse, cvs_rmse_mean = cvs_rmse_mean, tests_rmse = tests_rmse,
                              tests_accuracy = tests_accuracy, pipeline = pipeline, X_test = X_test,
                              y_test = y_test, cv = True):
    # Fit dataset in model
    model.fit(diamonds, labels)
    
    # Setup predictions
    predictions = model.predict(diamonds)
    
    # Get models performance
    model_mse = mean_squared_error(labels, predictions)
    model_rmse = np.sqrt(model_mse)
    
     # Cross validation
    cv_score = cross_val_score(model, diamonds, labels, scoring = "neg_mean_squared_error", cv = 10)
    cv_rmse = np.sqrt(-cv_score)
    cv_rmse_mean = cv_rmse.mean()
    
    print("RMSE: %.4f" %model_rmse)
    models_rmse.append(model_rmse)
    
    print("CV-RMSE: %.4f" %cv_rmse_mean)
    cvs_rmse_mean.append(cv_rmse_mean)
    
    print("--- Test Performance ---")
    
    X_test_prepared = pipeline.transform(X_test)
    
    # Fit test dataset in model
    model.fit(X_test_prepared, y_test)
    
    # Setup test predictions
    test_predictions = model.predict(X_test_prepared)
    
    # Get models performance on test
    test_model_mse = mean_squared_error(y_test, test_predictions)
    test_model_rmse = np.sqrt(test_model_mse)
    print("RMSE: %.4f" %test_model_rmse)
    tests_rmse.append(test_model_rmse)
    
    # Tests accuracy
    test_accuracy = round(model.score(X_test_prepared, y_test) * 100, 2)
    print("Accuracy:", str(test_accuracy)+"%")
    tests_accuracy.append(test_accuracy)
    
    # Check how well model works on Test set by comparing prices
    start = randint(1, len(y_test))
    some_data = X_test.iloc[start:start + 7]
    some_labels = y_test.iloc[start:start + 7]
    some_data_prepared = pipeline.transform(some_data)
    print("Predictions:\t", model.predict(some_data_prepared))
    print("Labels:\t\t", list(some_labels))
    
    models.append(model_name)
    plt.scatter(diamond_labels, model.predict(diamonds_ready))
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    x_lim = plt.xlim()
    y_lim = plt.ylim()
    plt.plot(x_lim, y_lim, "k--")
    plt.show()
    
    print("------- Test -------")
    plt.scatter(y_test, model.predict(X_test_prepared))
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.plot(x_lim, y_lim, "k--")
    plt.show()


# In[42]:


lin_reg = LinearRegression(normalize = True)
display_model_performance("Linear Regression", lin_reg)


# In[53]:


from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators = 10, random_state = 42)
display_model_performance("Random Forest Regression", forest_reg)


# In[78]:


compare_models = pd.DataFrame({ "Algorithms": models, "Models RMSE": models_rmse, "CV RMSE Mean": cvs_rmse_mean,
                              "Tests RMSE": tests_rmse, "Tests Accuracy": tests_accuracy })
compare_models.sort_values(by = "Tests Accuracy", ascending = False)


# In[ ]:





# In[101]:


import pickle

with open('final_model.pkl', 'wb') as f:
    pickle.dump(forest_reg, f)


# In[106]:


model= lin_reg


# In[103]:


new_diamond = [0.23, 5, 2, 1, 61.5, 55, 38.20,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]


# In[98]:


prediction = model.predict([new_diamond])[0]
print("\033[1m The market price of this new diamond is ${:.2f}".format(prediction))


# In[ ]:





# In[132]:


plt.scatter(dia['price'], dia['cut'])
plt.show()

# Divide the data to training set and test set
X_train, X_test, y_train, y_test = train_test_split(dia['price'], dia['cut'], test_size=0.20)


# In[133]:


# Creating the logistic regression model

# Helper function to normalize data
def normalize(X):
    return X - X.mean()

# Method to make predictions
def predict(X, b0, b1):
    return np.array([1 / (1 + exp(-1*b0 + -1*b1*x)) for x in X])

# Method to train the model
def logistic_regression(X, Y):

    X = normalize(X)

    # Initializing variables
    b0 = 0
    b1 = 0
    L = 0.001
    epochs = 300

    for epoch in range(epochs):
        y_pred = predict(X, b0, b1)
        D_b0 = -2 * sum((Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b0
        D_b1 = -2 * sum(X * (Y - y_pred) * y_pred * (1 - y_pred))  # Derivative of loss wrt b1
        # Update b0 and b1
        b0 = b0 - L * D_b0
        b1 = b1 - L * D_b1
    
    return b0, b1


# In[149]:


# Making predictions using scikit learn
from sklearn.linear_model import LogisticRegression

# Create an instance and fit the model 
lr_model = LogisticRegression()
lr_model.fit(X_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

# Making predictions
y_pred_sk = lr_model.predict(X_test.values.reshape(-1, 1))

plt.clf()
plt.scatter(X_test, y_test)
plt.scatter(X_test, y_pred_sk, c="red")
plt.show()

# Accuracy
print(f"Accuracy = {lr_model.score(X_test.values.reshape(-1, 1), y_test.values.reshape(-1, 1))}")


# In[135]:


accuracy_score(y_test,y_pred_sk)


# In[138]:


scaler=StandardScaler()


# In[142]:


x_train_S=scaler.fit_transform(X_train.values.reshape(-1, 1))
x_test_s=scaler.transform(X_test.values.reshape(-1, 1))


# In[152]:


lr_model.fit(x_train_S,y_train)


# In[147]:


x_train_S


# In[145]:


y_pred2=lr_model.predict(x_test_s)


# In[156]:


lr_model.score(x_test_s,y_pred_sk)

