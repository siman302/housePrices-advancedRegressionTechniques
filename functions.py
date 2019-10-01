import sys
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import percentile
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
import missingno as msno
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
np.set_printoptions(threshold=sys.maxsize)
import numpy as np
from scipy.stats import skew #for some statistics
from scipy import stats

def removeMissingValues(df_train):
	df_train['KitchenQual'].replace(np.nan, df_train['KitchenQual'].value_counts().idxmax(), inplace= True)	
	df_train['SaleType'].replace(np.nan, df_train['SaleType'].value_counts().idxmax(), inplace= True)	
	df_train['Exterior2nd'].replace(np.nan, df_train['Exterior2nd'].value_counts().idxmax(), inplace= True)	
	df_train['Exterior1st'].replace(np.nan, df_train['Exterior1st'].value_counts().idxmax(), inplace= True)	
	df_train['KitchenQual'].replace(np.nan, df_train['KitchenQual'].value_counts().idxmax(), inplace= True)			
	df_train['Electrical'].replace(np.nan, df_train['Electrical'].value_counts().idxmax(), inplace= True)	
	df_train['MSZoning'].replace(np.nan, df_train['MSZoning'].value_counts().idxmax(), inplace= True)
	df_train["Functional"] = df_train["Functional"].fillna("Typ")

	for col in ('GarageYrBlt', 'GarageArea', 'GarageCars','BsmtFinSF1', 'BsmtFinSF2', 'BsmtUnfSF','TotalBsmtSF', 'BsmtFullBath', 'BsmtHalfBath', 'MasVnrType', 'MasVnrArea'):
		df_train[col] = df_train[col].fillna(0)
	
	for col in ('MSSubClass', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2'):
		df_train[col] = df_train[col].fillna('None')
	
	df_train["LotFrontage"] = df_train.groupby("Neighborhood")["LotFrontage"].transform(lambda x: x.fillna(x.median()))
	return df_train

def missingValueGraphAndRelation(data):	
	try:
		plt.title(' Missing Values Graph ')
		sns.set_style("whitegrid")
		missing = data.isnull().sum()
		missing = missing[missing > 0]
		missing.sort_values(inplace=True)
		missing.plot.bar()	
		msno.heatmap(data) # relation of missing values with other values	
		print(data.isnull().sum().sort_values(ascending=False))
	except:
		print("No missing Value exit")

def replaceNaToNone(data):
	for col in ('Alley', 'BsmtQual', 'BsmtCond', 'BsmtExposure', 'BsmtFinType1', 'BsmtFinType2',
			  'FireplaceQu', 'GarageType', 'GarageFinish', 'GarageQual', 'GarageCond',
			   'PoolQC', 'Fence', 'MiscFeature'):
		data[col] = data[col].fillna('None')
	return data

def checkDistribution(df_train):	

	plt.figure(0); plt.title('Histogram')
	plt.hist(df_train,bins=100)	

	plt.figure(1); plt.title('Johnson SU')
	sns.distplot(df_train, fit=stats.johnsonsu)
	
	plt.figure(2); plt.title('Normal')
	sns.distplot(df_train, fit=stats.norm)

	plt.figure(3); plt.title('Log Normal')
	sns.distplot(df_train, fit=stats.lognorm)
	
	# Get also the QQ-plot
	plt.figure()
	stats.probplot(df_train, plot=plt)
	plt.show()
	print(skew(df_train)) # +ve value tell it is right skew and viceversa


def outliergraph(df_train,IndeColoum,deColoum):
	sns.boxplot(x=df_train[IndeColoum])
	fig, ax = plt.subplots()
	ax.scatter(x = df_train[IndeColoum], y = df_train[deColoum])
	plt.ylabel(deColoum, fontsize=13)
	plt.xlabel(IndeColoum, fontsize=13)
	plt.show()	
	plt.style.use('ggplot')
	plt.hist(df_train[IndeColoum], bins=60)

#https://stackoverflow.com/questions/43588679/issue-with-onehotencoder-for-categorical-features
def handleCatagoricalData(train):
	
	train.columns
	pd.get_dummies(train, prefix=['country'], drop_first=True)

	obj =  LabelEncoder()	
	makecoloums_State = OneHotEncoder(categories='auto')	
	
	temp = np.concatenate((
	makecoloums_State.fit_transform(obj.fit_transform(train[:,0]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,1]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,4]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,5]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,6]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,7]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,8]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,9]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,10]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,11]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,12]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,13]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,14]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,15]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,16]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,17]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,20]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,21]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,22]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,23]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,24]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,26]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,27]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,28]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,29]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,30]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,31]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,32]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,34]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,38]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,37]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,39]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,40]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,41]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,52]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,54]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,56]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,57]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,59]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,62]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,63]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,64]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,71]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,72]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,73]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,77]).reshape(-1, 1)).toarray()[:,1:],
	makecoloums_State.fit_transform(obj.fit_transform(train[:,78]).reshape(-1, 1)).toarray()[:,1:],
	),axis=1)
	
	return temp

def standardScaler(train, test):
	sc = StandardScaler()
	train = sc.fit_transform(train)
	test = sc.transform(test)
	return train, test

def dataframeToArray(df_trainY, df_testY):	
#	X_train = df_train.iloc[:,:].values
	Y_train = df_trainY.iloc[:,:].values
#	X_test = df_test.iloc[:,:].values
	Y_test = df_testY.iloc[:,1].values
	return Y_train, Y_test

def confusionMatrix(Y_test, Y_pred):
	cm = confusion_matrix(Y_test, Y_pred)
	print(cm)
	return cm

def replaceNA(df_train,df_test):	
	col = df_train.select_dtypes(np.object).columns
	print(col)
	df_train[col] = df_train[col].fillna("None")
	df_test[col] = df_test[col].fillna("None")
	return df_train,df_test

def missingValueHandler(train, test, div):
	train = train[train[:,div].argsort()]
	test = test[test[:,div].argsort()]
	start_index = 0
	end_index = 0
	variableList = train[:,div:div+1]
	unique, counts = np.unique(variableList, return_counts=True)
	for (index, item) in enumerate(counts):
		start_index = end_index
		end_index = start_index + int(item)	
		train = removeImpurties(train, start_index, end_index)
		test = removeImpurties(test, start_index, end_index)
	return train, test

def splitTheData(train, test):
	X_train = train[:,:-1]
	Y_train = train[:,-1:]
	X_test = test[:,:-1]
	Y_test = test[:,-1:]
	return X_train, Y_train, X_test, Y_test

def removeImpurties(cat,noncat):	
	imputerformean,imputerformedian,imputerforfrequency = imputerStrategys(0)	
	
	imputermedian = imputerformedian.fit(cat)
	cat = imputermedian.transform(cat)
	imputerfrequency = imputerforfrequency.fit(noncat)
	noncat = imputerfrequency.transform(noncat)
	X = np.hstack((cat, noncat))
	return X

def removeImpurtiesDataframe(df_train,df_test):
	
	return df_train,df_test

def imputerStrategys(axis = 0):
	imputerformean = Imputer(missing_values = 'NaN', strategy = 'mean', axis = axis)
	imputerformedian = Imputer(missing_values = 'NaN', strategy = 'median', axis = axis)
	imputerforfrequency = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = axis)
	return imputerformean,imputerformedian,imputerforfrequency

def dataframeToArray1(df_train,df_test,df_real):	
	X_train = df_train.iloc[:,1:].values
	X_test = df_test.iloc[:,1:].values
	Y_test = df_real.iloc[:,1:].values
	X_test = np.hstack((X_test, Y_test)) # attach both of them
	return X_train, X_test

def getDatasets():
	df_train = pd.read_csv('train.csv')
	df_test = pd.read_csv('test.csv')
	df_real = pd.read_csv('sample_submission.csv')
	return df_train,df_test,df_real

def getAllHeaders(dataset):
	myList = dataset.columns.tolist();
	print(myList)
	return myList

def getColoumIndex(dataset, col):
	return dataset.columns.get_loc(col), dataset[col]
	
def getOutliersPercentile(dataset):
	# calculate interquartile range
	q25, q75 = percentile(dataset, 25), percentile(dataset, 75)
	iqr = q75 - q25
	print('Percentiles: 25th=%.3f, 75th=%.3f, IQR=%.3f' % (q25, q75, iqr))
	# calculate the outlier cutoff
	cut_off = iqr * 1.5
	lower, upper = q25 - cut_off, q75 + cut_off
	# identify outliers
	outliers = [x for x in dataset if x < lower or x > upper]
	print('Identified outliers: %d' % len(outliers))
	# remove outliers
	outliers_removed = [x for x in dataset if x >= lower and x <= upper]
	print('Non-outlier observations: %d' % len(outliers_removed))

def checkOutliers(dataset, hist):
	if hist:
		plt.style.use('ggplot')
		plt.hist(dataset, bins=60)
	else:
		sns.boxplot(x=dataset)

def correlationMatrix(dataset, dependentVariable, allVariables = True, k=3):
	corrmat = dataset.corr()
	f, ax = plt.subplots(figsize=(15, 12))	
	if allVariables:		
		sns.heatmap(corrmat, vmax=.3, square=False, cmap='BuGn');
	else:
		cols = corrmat.nlargest(k, dependentVariable)[dependentVariable].index
		print(cols)
		cm = np.corrcoef(dataset[cols].values.T)
		sns.set(font_scale=1)
		sns.heatmap(cm, cbar=True, annot=True, square=False, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)
	
def correlationMatrix2Variables(dataset, dependentVariable, V2):
	corrmat = dataset.corr()
	cols = corrmat.nlargest(5, dependentVariable)[dependentVariable].index
	V2 = pd.Index([V2]) 
	cm = np.corrcoef(dataset[cols].values.T)
	sns.set(font_scale=1)
	sns.heatmap(cm, cbar=True, annot=True, square=False, fmt='.2f', annot_kws={'size': 8}, yticklabels=cols.values, xticklabels=cols.values)

def RemoveOutliers(dataset):
	mean = np.mean(dataset, axis=0)
	sd = np.std(dataset, axis=0)
	#df_train[(df_train['GrLivArea']>4000) & (df_train['GrLivArea']<300000)].index
	dataset = [x for x in dataset if (x > mean - 2 * sd)]
	dataset = [x for x in dataset if (x < mean + 2 * sd)]
	return dataset

