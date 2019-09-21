import sys
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import percentile
from sklearn.preprocessing import Imputer
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
np.set_printoptions(threshold=sys.maxsize)


#https://stackoverflow.com/questions/43588679/issue-with-onehotencoder-for-categorical-features
def handleCatagoricalData(train):
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
	print("original size : ",train.shape)
	train = np.delete(train,[0,1,4,5,6,7,8,9,10,11,12,13,14,15,
						  16,17,20,21,22,23,24,26,27,28,29,30,39,
						  31,32,34,38,37,40,41,52,54,56,57,59,
						  62,63,64,71,72,73,77,78],axis=1)
	print("After Remove Coloum : ",train.shape)
	print("Extended Coloum : ",temp.shape)
	train = np.concatenate((train,temp),axis=1)
	print("After Join Coloum : ",train.shape)
	print(" : - - - - - - - - - - - - - - - - - : ")
	return train

def standardScaler(train, test):
	sc = StandardScaler()
	train = sc.fit_transform(train)
	test = sc.transform(test)
	return train, test

def confusionMatrix(Y_test, Y_pred):
	cm = confusion_matrix(Y_test, Y_pred)
	print(cm)
	return cm

def replaceNA(df_train,df_test):	
	c = df_train.select_dtypes(np.object).columns
	df_train[c] = df_train[c].fillna("ReplaceNA")
	df_test[c] = df_test[c].fillna("ReplaceNA")
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

#	getOutliersPercentile(dataChunk)
#	checkOutliers(dataChunk, True)
def removeImpurties(data, start_index, end_index):	
	imputerformean,imputerformedian,imputerforfrequency = imputerStrategys(0)	
	imputermedian = imputerformedian.fit(data[start_index:end_index,2:3])
	data[start_index:end_index,2:3] = imputermedian.transform(data[start_index:end_index,2:3])
	return data

def imputerStrategys(axis = 0):
	imputerformean = Imputer(missing_values = 'NaN', strategy = 'mean', axis = axis)
	imputerformedian = Imputer(missing_values = 'NaN', strategy = 'median', axis = axis)
	imputerforfrequency = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = axis)
	return imputerformean,imputerformedian,imputerforfrequency

def dataframeToArray(df_train,df_test,df_real):	
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
	if allVariables:		
		f, ax = plt.subplots(figsize=(12, 9))
		sns.heatmap(corrmat, vmax=.8, square=True);
	else:
		cols = corrmat.nlargest(k, dependentVariable)[dependentVariable].index
		cm = np.corrcoef(dataset[cols].values.T)
		sns.set(font_scale=1)
		sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size': 10}, yticklabels=cols.values, xticklabels=cols.values)
	
def RemoveOutliers(dataset):
	mean = np.mean(dataset, axis=0)
	sd = np.std(dataset, axis=0)
	dataset = [x for x in dataset if (x > mean - 2 * sd)]
	dataset = [x for x in dataset if (x < mean + 2 * sd)]
	return dataset