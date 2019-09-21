
from sklearn.metrics import mean_squared_error
from math import sqrt
try:
	from functions import removeImpurties, dataframeToArray, replaceNA, getDatasets, correlationMatrix
	from algorithm import linearRegression
except:
    print("HeaderFiles not in access")
    raise SystemExit    


# Import dataset
df_train,df_test,df_testY = getDatasets()

# Get Info About dataset
df_train.info()
df_train.info()
df_test.info()

df_train.head(2)
df_test.head(2)

df_train.isnull().values.sum()
df_train.notnull().values.sum()
df_test.isnull().values.sum()

df_train.isnull().sum().sort_values(ascending=False)
df_test.isnull().sum().sort_values(ascending=False)

df_train.dtypes
df_test.dtypes

# Analysis
correlationMatrix(df_train, "SalePrice", False, 15)

# change the NA to removeNA to not confuse with null
df_train,df_test = replaceNA(df_train,df_test)
df_train.isnull().values.sum()
df_test.isnull().values.sum()

# drop the unreleated coloums below the 0.3
df_trainY = df_train[['SalePrice']]
df_train = df_train[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea','TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces','BsmtFinSF1']]
df_test = df_test[['OverallQual', 'GrLivArea', 'GarageCars', 'GarageArea','TotalBsmtSF', '1stFlrSF', 'FullBath', 'TotRmsAbvGrd', 'YearBuilt','YearRemodAdd', 'GarageYrBlt', 'MasVnrArea', 'Fireplaces','BsmtFinSF1']]


# Sparate the cat vs noncat
#cat_df_train = df_train[df_train.select_dtypes(include=[np.object]).columns].head(df_train.shape[0])
#noncat_df_train = df_train[df_train.select_dtypes(exclude=[np.object]).columns].head(df_train.shape[0])
#cat_df_test = df_test[df_test.select_dtypes(include=[np.object]).columns].head(df_train.shape[0])
#noncat_df_test = df_test[df_test.select_dtypes(exclude=[np.object]).columns].head(df_train.shape[0])
#cat_df_train.isnull().values.sum()
#noncat_df_train.isnull().values.sum()
#cat_df_test.isnull().values.sum()
#noncat_df_test.isnull().values.sum()
	# handle the catagorical variable
	# Way 001 : when you don't have all the vales
#cat_df_train["MSZoning"].value_counts()
#pd.get_dummies(cat_df_train['MSZoning'],prefix=['MSZoning'])
#def_train = handleCatagoricalData(cat_df_train)	
	# Way 002 : when you kn0w the coloums
#cat_df_test.columns
#pd.get_dummies(cat_df_train['MSZoning'].astype('category',categories=["A","C","FV","I","RH","RL","RP","RM"]),prefix=[cat_df_train.columns[0]],drop_first=True).shape


# Sparate the cat vs noncat
cat_df_train = df_train[['OverallQual','YearBuilt','YearRemodAdd','GarageYrBlt']]
noncat_df_train = df_train[['GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','MasVnrArea','Fireplaces','BsmtFinSF1']]

cat_df_test = df_test[['OverallQual','YearBuilt','YearRemodAdd','GarageYrBlt']]
noncat_df_test = df_test[['GrLivArea','GarageCars','GarageArea','TotalBsmtSF','1stFlrSF','FullBath','TotRmsAbvGrd','MasVnrArea','Fireplaces','BsmtFinSF1']]

# Remove null values by frequency & median
cat_df_train.isnull().values.sum()
noncat_df_train.isnull().values.sum()

cat_df_test.isnull().values.sum()
noncat_df_test.isnull().values.sum()

X_train = removeImpurties(cat_df_train,noncat_df_train)
X_test = removeImpurties(cat_df_test,noncat_df_test)


Y_train, Y_test = dataframeToArray(df_trainY, df_testY)

#	getOutliersPercentile(dataChunk)
#	checkOutliers(dataChunk, True)
# apply the linearRegression
Y_predict = linearRegression(X_train, Y_train, X_test)

# RMSE
rms = sqrt(mean_squared_error(Y_test, Y_predict))
print(rms)


#import numpy as np
#from sklearn.metrics import mean_squared_error
#from math import sqrt

# important link
# https://www.geeksforgeeks.org/python-replace-nan-values-with-average-of-columns/
# https://www.datacamp.com/community/tutorials/categorical-data