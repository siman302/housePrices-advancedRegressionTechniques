
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


try:
	from functions import removeMissingValues, missingValueGraph, replaceNaToNone, checkDistribution, outliergraph, getDatasets, correlationMatrix
	from algorithm import AlgoXgboost
except:
    print("HeaderFiles not in access")
    raise SystemExit    


# Import dataset
df_train,df_test,df_testY = getDatasets()


# Get Info About dataset
df_train.columns

df_train.info()
df_test.info()

df_train.head(2)
df_test.head(2)

df_train.dtypes
df_test.dtypes

# Convert some coloums into the Object because they are the catagorical objects
# Remove Outliners only in non-catagorical dataset
# Remember : Apply only on the df_train dataset and 
# Note : OverallQual, OverallCond there are catagorical variables too 
df_train["MSSubClass"] = df_train["MSSubClass"].astype(object) 
df_train["MSSubClass"] = df_train["MSSubClass"].astype(object) 

df_train.select_dtypes(exclude=[np.object]).columns
temp = df_train[df_train.select_dtypes(exclude=[np.object]).columns].head(2)
del temp

df_train.shape
outliergraph(df_train,'LotFrontage','SalePrice')
df_train = df_train.drop(df_train[(df_train['LotFrontage']>250)].index)
df_train = df_train.drop(df_train[(df_train['LotArea']>150000)].index)
df_train = df_train.drop(df_train[(df_train['MasVnrArea']>1200)].index)
df_train = df_train.drop(df_train[(df_train['BsmtFinSF1']>2000)].index)
df_train = df_train.drop(df_train[(df_train['BsmtFinSF2']>1200)].index)
df_train = df_train.drop(df_train[(df_train['BsmtUnfSF']>2200)].index)
df_train = df_train.drop(df_train[(df_train['1stFlrSF']>3000)].index)
df_train = df_train.drop(df_train[(df_train['1stFlrSF']>3000)].index)
df_train = df_train.drop(df_train[(df_train['2ndFlrSF']>1750)].index)
df_train = df_train.drop(df_train[(df_train['LowQualFinSF']>550)].index)
df_train = df_train.drop(df_train[(df_train['GrLivArea']>4000) & (df_train['GrLivArea']<300000)].index)
df_train = df_train.drop(df_train[(df_train['GarageArea']>1200)].index)
df_train = df_train.drop(df_train[(df_train['OpenPorchSF']>800)].index)
df_train = df_train.drop(df_train[(df_train['OpenPorchSF']>500)].index)
df_train = df_train.drop(df_train[(df_train['EnclosedPorch']>500)].index)
df_train = df_train.drop(df_train[(df_train['MiscVal']>3000)].index)
df_train = df_train.drop(df_train[(df_train['SalePrice']>700000)].index)
df_train.shape # 27 rows drops


# Combine the test and test result
df_testY.rename(columns = {'Id': 'Tid'}, inplace = True)
df_test = pd.concat([df_test, df_testY], axis=1, sort=False)
del df_testY


# check the ids
print(df_test['Id'].equals(df_test['Tid']))
print(df_test[['Id','Tid']])
df_test.drop("Tid", axis = 1, inplace = True)

# Combine the datasets
dataset = pd.concat([df_train, df_test])
print(df_train.shape)
print(df_test.shape)
print(dataset.shape)
dataset = pd.concat([df_train, df_test], sort=False)

# Save the Id column
train_Ids = df_train['Id']
test_Ids = df_test['Id']
print(train_Ids.size)
print(test_Ids.size)
del df_train
del df_test


# drop the id column of a dataset
dataset.drop("Id", axis = 1, inplace = True)
dataset.shape


# Our dependent Variable Analysis
dataset['SalePrice'].describe()
plt.hist(dataset['SalePrice'],bins=100)
	# make it a more normally distributed if not
	# beacuse (linear) models love normally distributed data
checkDistribution(dataset['SalePrice']) # graph show it is right skew
dataset["SalePrice"] = np.log1p(dataset["SalePrice"])  # apply log(1+x) to make it normal distribution
checkDistribution(dataset['SalePrice']) # graph show it is right skew
plt.hist(dataset['SalePrice'],bins=100)


# replace the NA to the None to eliminate the confile
missingValueGraph(dataset)
dataset = replaceNaToNone(dataset)
missingValueGraph(dataset)


# Analysis the featuers or relation of dependent and independent variables
correlationMatrix(dataset, "SalePrice")
correlationMatrix(dataset, "SalePrice", False, 15)


# remove missing values from catagorical and non catagorical variables
missingValueGraph(dataset)
dataset = removeMissingValues(dataset)
missingValueGraph(dataset)


# Handle Catagorical Variables and avoid dummy variable trap
dataset = pd.get_dummies(dataset, drop_first=True)


# Handle Non-catagorical Variables
# do it later
#nonCat_df_train['TotalSF'] = nonCat_df_train['TotalBsmtSF'] + nonCat_df_train['1stFlrSF'] + nonCat_df_train['2ndFlrSF']
#nonCat_df_train.drop("TotalBsmtSF", axis = 1, inplace = True)
#nonCat_df_train.drop("1stFlrSF", axis = 1, inplace = True)
#nonCat_df_train.drop("2ndFlrSF", axis = 1, inplace = True)

# Check the skew of all numerical features
#skewed_feats = nonCat_df_train.skew(axis = 0, skipna = True).sort_values(ascending=False)
#skewness = pd.DataFrame({'Skew' :skewed_feats})
#skewness = skewness[abs(skewness) > 0.75]
#print("numerical features to Box Cox transform count".format(skewness.shape[0]))

## transform non-normal dependent variables into a normal shape
#from scipy.special import boxcox1p
#skewed_features = skewness.index
#lam = 0.15
#nonCat_df_train_skew = pd.DataFrame() 
#for feat in skewed_features:
#    nonCat_df_train_skew[feat] = boxcox1p(nonCat_df_train[feat], lam)


# Split the dataset into test and train
df_train = dataset[0:1456-23]
df_test = dataset[1456-23:]

df_trainY = df_train['SalePrice']
df_testY = df_test['SalePrice']

df_train.drop("SalePrice", axis = 1, inplace = True)
df_test.drop("SalePrice", axis = 1, inplace = True)

Xg = AlgoXgboost(df_train, df_trainY, df_test, df_testY)
print(Xg)


