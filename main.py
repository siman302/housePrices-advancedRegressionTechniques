
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt
try:
	from functions import handleCatagoricalData, replaceNA, splitTheData, dataframeToArray, getColoumIndex, missingValueHandler, getDatasets, correlationMatrix
	from algorithm import linearRegression
except:
    print("HeaderFiles not in access")
    raise SystemExit    

# Import dataset
df_train,df_test,df_real = getDatasets()

# Info About dataset
#print(df_train.info())
#df_train.boxplot('dep_time','origin',rot = 30,figsize=(5,6))
#cat_df_flights = df_flights.select_dtypes(include=['object']).copy()
#cat_df_flights.head()
#print(cat_df_flights.isnull().values.sum())
#print(cat_df_flights.isnull().sum())
#print(df_test.info())
#print(df_real.info())
#print (df_train.dtypes)

# Analysis
correlationMatrix(df_train, "SalePrice", False, 3)

# OverallQual = 0.79, relate most
index, chunk = getColoumIndex(df_train, "OverallQual")

# change the NA to removeNA to not confuse with null
df_train,df_test = replaceNA(df_train,df_test)

# Dataframe To Arrays and remove Id's
X_train, X_test = dataframeToArray(df_train,df_test,df_real)

# Sort on the base of coloum values OverallQual
# -1 due to remove id coloum
X_train,X_test = missingValueHandler(X_train,X_test,index-1)

# split the dataset on Dependent and Independent Variable
X_train, Y_train, X_test, Y_test = splitTheData(X_train,X_test)

# handleCatagoricalData
X_train = handleCatagoricalData(X_train)
X_test = handleCatagoricalData(X_test)
X_test = X_test[:,:-5] # reset the variable

# remove the extra 'nan'
# https://www.geeksforgeeks.org/python-replace-nan-values-with-average-of-columns/
X_train = np.where(np.isnan(X_train.astype(float)), np.ma.array(X_train.astype(float), mask = np.isnan(X_train.astype(float))).mean(axis = 0), X_train.astype(float))    
X_test = np.where(np.isnan(X_test.astype(float)), np.ma.array(X_test.astype(float), mask = np.isnan(X_test.astype(float))).mean(axis = 0), X_test.astype(float))    

# apply the linearRegression
Y_predict = linearRegression(X_train, Y_train, X_test)

# RMSE
rms = sqrt(mean_squared_error(Y_test, Y_predict))
print(rms)
