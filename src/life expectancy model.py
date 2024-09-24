######################################
#Import Key Libraries and Datasets
######################################
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

# read the csv file 
df = pd.read_csv('Life_Expectancy_Data.csv')
df

# Check the dataframe info
df.info()

#Gives statistical summary of data
df.describe()

######################################
#Obtain dataframe statistical summary
######################################
#All of the maximum values of the dataframe
df.max()

#To check the stats of a specific column. Be sure the column name is an exact match
df[("Life expectancy ")].max()

#All the minimum values
df.min()

#The average values for all the columns
df.mean()

######################################
#Perform Data Visualization
######################################

# check if there are any Null values. Any white cell indicates a number in a cell. 
#A blue cell indicates an empty cell. 
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

# Plot the histogram
df.hist(bins = 30, figsize = (20, 20), color = 'r');

# Plot the correlation matrix
plt.figure(figsize = (20,20))
corr_matrix = df.corr() # Calculates correlation matrix
sns.heatmap(corr_matrix, annot = True) #Use sns.heatmap to plot correlaation matrix. Annotate adds numbers, while = False would remove the,
plt.show()



######################################
#Perform Feature Engineering
######################################
# Perform one-hot encoding, which means to encode my features('Status' column) and convert it to binary (1's and 0's) because we cant feed text data into models
df = pd.get_dummies(df, columns = ['Status'])

#Sum up all the missing values
df.isnull().sum()

#Now depending on your project, the best approach isn't always to remove all the rows that have missing values because you might miss very important data
#So in this approach we will fill those empty slots with the average of that type to account for the missing elements

# Check the number of null values for the columns having null values
df.isnull().sum()[np.where(df.isnull().sum() != 0)[0]]

# Since most of the are continous values we fill them with mean
df = df.apply(lambda x: x.fillna(x.mean()),axis=0)

# Check the number of null values again
df.isnull().sum()[np.where(df.isnull().sum() != 0)[0]]
#You should see that the output now "Series([], dtype: int64)" shows no null elements

# Now we can run the heatmap of the check if there are any Null values
sns.heatmap(df.isnull(), yticklabels = False, cbar = False, cmap="Blues")

#Now we are going to split the data into input and output
# Create train and test data
X = df.drop(columns = ['Life expectancy '])
y = df[['Life expectancy ']]

# Convert the data type to float32
X = np.array(X).astype('float32')
y = np.array(y).astype('float32')

# spliting the data into training, testing and validation sets and using 70%% for training and 30% for testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)

#The output of this shows that the training data will use 2056 elements of the dataset
X_train.shape

#The output of this shows that the testing data will use 882 elements of the dataset
X_test.shape



######################################
#Train an XG-BOOST Regression Model
######################################

#Will have to install xg boost 
!pip install xgboost

import xgboost as xgb

# Train an XGBoost regressor model 
#in this code we use squared error becaause xgboost learns from its mistakes using mean error, the max depth refers to the max amount of learning tree models in the xgboost cycle, then how many models/estimators
model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 30, n_estimators = 100)

#Once this code is ran, the model is trained
model.fit(X_train, y_train)


######################################
#Evaluate Trained Models Performance
######################################

#Now we can take the model and apply it to testing data

# predict the score of the trained model using the testing dataset
result = model.score(X_test, y_test)
print("Accuracy : {}".format(result))

# make predictions on the test data
y_predict = model.predict(X_test)
y_predict #Will ouput the predictions coming from the model from every sample from the testing dataset
#Now we can compare what the model has predicted vs. the ground-truth data, which would be in y-test


#So now we need to import r2, meansquared, meanabsolute, and sqrt
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt

k = X_test.shape[1]
n = len(X_test)

#Calculate root meansquarederror, the mse, mae and r2 comparing true data vs perdictions
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2)


######################################
#Optional - Set the max depth hyperparameter to a very small number and retrain the model to make inferences on results
######################################

# Train an XGBoost regressor model and changed the max_depth to 2
import xgboost as xgb
model = xgb.XGBRegressor(objective ='reg:squarederror', learning_rate = 0.1, max_depth = 2, n_estimators = 100)
model.fit(X_train, y_train)

# predict the score of the trained model using the testing dataset
result = model.score(X_test, y_test)
print("Accuracy : {}".format(result))

# make predictions on the test data
y_predict = model.predict(X_test)

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from math import sqrt
k = X_test.shape[1]
n = len(X_test)
RMSE = float(format(np.sqrt(mean_squared_error(y_test, y_predict)),'.3f'))
MSE = mean_squared_error(y_test, y_predict)
MAE = mean_absolute_error(y_test, y_predict)
r2 = r2_score(y_test, y_predict)
adj_r2 = 1-(1-r2)*(n-1)/(n-k-1)

print('RMSE =',RMSE, '\nMSE =',MSE, '\nMAE =',MAE, '\nR2 =', r2, '\nAdjusted R2 =', adj_r2) 

#Accuracy of the model is significantly less
#RMSE Increased
#So you can see that max_depths adds more learning cycles for the model

