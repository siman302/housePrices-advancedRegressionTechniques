
import numpy as np
import pandas as pd
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


try:
	from functions import removeMissingValues, missingValueGraphAndRelation, replaceNaToNone, checkDistribution, outliergraph, getDatasets, correlationMatrix
	from algorithm import AlgoSCR, AlgoGBR, AlgoXgboost, AlgoRForest, AlgoSVR, AlgoLGBM, AlgoENet, AlgoLasso, AlgoKRR, Kernal
except:
    print("HeaderFiles not in access")
    raise SystemExit    


# Import dataset
df_train,df_test,df_testY = getDatasets()

# Get Info About dataset
# 23 nominal, 23 ordinal
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
df_train["OverallQual"] = df_train["OverallQual"].astype(object)
df_train['OverallCond'] = df_train['OverallCond'].astype(object)
df_train.select_dtypes(exclude=[np.object]).columns
temp = df_train[df_train.select_dtypes(exclude=[np.object]).columns].head(2)
del temp


# Remove Outliners(5 outliers exit) 
df_train.shape
#outliergraph(df_train,'GrLivArea','SalePrice')
df_train = df_train[df_train.GrLivArea < 4000]


# Our dependent Variable Analysis
# make it a more normally distributed if not beacuse (linear) models love normally distributed data
#checkDistribution(df_train['SalePrice']) # graph show it is right skew
df_train["SalePrice"] = np.log(df_train["SalePrice"])  # apply log(1+x) to make it normal distribution
#checkDistribution(df_train['SalePrice']) # graph show it is right skew


# Combine the test and test result
df_testY.rename(columns = {'Id': 'Tid'}, inplace = True)
df_test = pd.concat([df_test, df_testY], axis=1, sort=False)
df_trainY = df_train["SalePrice"]
df_testY = df_testY["SalePrice"]


# check the ids
print(df_test['Id'].equals(df_test['Tid']))
print(df_test[['Id','Tid']])
df_test.drop("Tid", axis = 1, inplace = True)


# Combine the datasets
print(df_train.shape)
print(df_test.shape)
dataset = pd.concat([df_train, df_test], sort=False)
print(dataset.shape)


# Analysis the featuers or relation of dependent and independent variables
#correlationMatrix(dataset, "SalePrice")
#correlationMatrix(dataset, "SalePrice", False)


# drop the id column of a dataset
dataset.drop("SalePrice", axis = 1, inplace = True)
dataset.drop("Id", axis = 1, inplace = True)
dataset.shape


# replace the NA to the None to eliminate the confile
#missingValueGraphAndRelation(dataset)
dataset = replaceNaToNone(dataset)
#missingValueGraphAndRelation(dataset)


# remove missing values from catagorical and non catagorical variables
#missingValueGraphAndRelation(dataset)
dataset = removeMissingValues(dataset)
#missingValueGraphAndRelation(dataset)


# Remove the useless coloums
import matplotlib.pyplot as plt
plt.scatter(df_train['Utilities'], df_train['SalePrice']) #only 1 missing value
df_train = df_train.drop(['Utilities'], axis=1)


# Handle Non-catagorical Variables
dataset['TotalSF'] = dataset['TotalBsmtSF'] + dataset['1stFlrSF'] + dataset['2ndFlrSF']

# Normalized there dataset
from scipy.stats import skew #for some statistics
numeric_feats = dataset.dtypes[dataset.dtypes != "object"].index
skewness = dataset[numeric_feats].apply(lambda x: skew(x.dropna())).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewness})
skewness = skewness[abs(skewness) > 0.75]

from scipy.special import boxcox1p
skewed_features = skewness.index
for feat in skewed_features:
    dataset[feat] = boxcox1p(dataset[feat], 0.15)
del feat
del skewness

# Handle Catagorical Variables and avoid dummy variable trap
dataset = pd.get_dummies(dataset, drop_first=True)

# Split the dataset into test and train
df_train = dataset[0:1456]
df_test = dataset[1456:]


## train and predict on the same dataset
krr = AlgoKRR(df_train, df_trainY) #
gbr = AlgoGBR(df_train, df_trainY) #
lgb = AlgoLGBM(df_train, df_trainY) # 1
net = AlgoENet(df_train, df_trainY) #
svr = AlgoSVR(df_train, df_trainY)  #
xgb = AlgoXgboost(df_train, df_trainY) # 3
rF = AlgoRForest(df_train, df_trainY) # 2
lasso = AlgoLasso(df_train, df_trainY) #

# Kernal
predict = Kernal(df_train, df_trainY, df_test, df_testY, lgb, rF, xgb)
df_testY['SalePrice'] = predict
df_testY.to_csv("SubmitResults.csv", sep=',', index=False)

