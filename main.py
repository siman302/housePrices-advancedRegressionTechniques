
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


try:
	from functions import removeMissingValues, missingValueGraph, replaceNaToNone, checkDistribution, outliergraph, getDatasets, correlationMatrix
except:
    print("HeaderFiles not in access")
    raise SystemExit    

# Import dataset
df_train,df_test,df_testY = getDatasets()


# Get Info About dataset
df_train.columns

df_train.info()
df_train.info()
df_test.info()

df_train.head(2)
df_test.head(2)

df_train.dtypes
df_test.dtypes


#Save the 'Id' column
train_Ids = df_train['Id']
test_Ids = df_test['Id']
df_train.drop("Id", axis = 1, inplace = True)
df_test.drop("Id", axis = 1, inplace = True)


# Remove Outliners only in non catagorical data
# only from train dataset
df_train.select_dtypes(exclude=[np.object]).columns
df_train[df_train.select_dtypes(exclude=[np.object]).columns].head(2)
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


# Our dependent Variable Analysis
df_train['SalePrice'].describe()
df_testY['SalePrice'].describe()
sns.set()
plt.hist(df_train['SalePrice'],bins=100)
plt.show()
sns.set()
plt.hist(df_testY['SalePrice'],bins=100)
plt.show()
	# make it a more normally distributed if not
	# beacuse (linear) models love normally distributed data
checkDistribution(df_train['SalePrice']) # graph show it is right skew
df_train["SalePrice"] = np.log1p(df_train["SalePrice"])  # apply log(1+x) to make it normal
checkDistribution(df_train['SalePrice'])


# replace the NA to the None to eliminate the confile
missingValueGraph(df_test)
df_train = replaceNaToNone(df_train)
df_test = replaceNaToNone(df_test)
missingValueGraph(df_test)


# Analysis the featuers or relation of dependent and independent variables
correlationMatrix(df_train, "SalePrice", False, 10)


# remove missing values from catagorical and non catagorical variables
missingValueGraph(df_train)
df_train = removeMissingValues(df_train)
df_test = removeMissingValues(df_test)
missingValueGraph(df_train)


# Split the catagorical and non catagorical data
df_train["MSSubClass"]=df_train["MSSubClass"].astype(object) 
#df_train["OverallQual"]=df_train["OverallQual"].astype(object) take it as a cat_df_train
#df_train["OverallCond"]=df_train["OverallCond"].astype(object) take it as a cat_df_train
cat_df_train = df_train[df_train.select_dtypes(include=[np.object]).columns].head(df_train.shape[0])
nonCat_df_train = df_train[df_train.select_dtypes(exclude=[np.object]).columns].head(df_train.shape[0])


# Handle Catagorical Variables and avoid dummy variable trap
cat_df_train = pd.get_dummies(cat_df_train, drop_first=True)


# Handle Non-catagorical Variables
# do it later
#nonCat_df_train['TotalSF'] = nonCat_df_train['TotalBsmtSF'] + nonCat_df_train['1stFlrSF'] + nonCat_df_train['2ndFlrSF']
#nonCat_df_train.drop("TotalBsmtSF", axis = 1, inplace = True)
#nonCat_df_train.drop("1stFlrSF", axis = 1, inplace = True)
#nonCat_df_train.drop("2ndFlrSF", axis = 1, inplace = True)

# Check the skew of all numerical features
skewed_feats = nonCat_df_train.skew(axis = 0, skipna = True).sort_values(ascending=False)
skewness = pd.DataFrame({'Skew' :skewed_feats})
skewness = skewness[abs(skewness) > 0.75]
print("numerical features to Box Cox transform count".format(skewness.shape[0]))

## transform non-normal dependent variables into a normal shape
#from scipy.special import boxcox1p
#skewed_features = skewness.index
#lam = 0.15
#nonCat_df_train_skew = pd.DataFrame() 
#for feat in skewed_features:
#    nonCat_df_train_skew[feat] = boxcox1p(nonCat_df_train[feat], lam)

   






