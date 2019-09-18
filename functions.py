import sys
sys.setrecursionlimit(100000) #Increase the recursion limit of the OS
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import percentile
from sklearn.preprocessing import Imputer

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
		removeImpurties(train, start_index, end_index)
		break
	return train, test

def removeImpurties(train, start_index, end_index):
	
	imputerformean,imputerformedian,imputerforfrequency = imputerStrategys(0)	
#	getOutliersPercentile(dataChunk)
#	checkOutliers(dataChunk, True)

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,:2])
	train[start_index:end_index,:2] = imputerfrequency.transform(train[start_index:end_index,:2])

	imputermedian = imputerformedian.fit(train[start_index:end_index,2:4])
	train[start_index:end_index,2:4] = imputermedian.transform(train[start_index:end_index,2:4])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,4:16])
	train[start_index:end_index,4:16] = imputerfrequency.transform(train[start_index:end_index,4:16])

	imputermedian = imputerformedian.fit(train[start_index:end_index,16:20])
	train[start_index:end_index,16:20] = imputermedian.transform(train[start_index:end_index,16:20])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,20:25])
	train[start_index:end_index,20:25] = imputerfrequency.transform(train[start_index:end_index,20:25])

	imputermedian = imputerformedian.fit(train[start_index:end_index,25:26])
	train[start_index:end_index,25:26] = imputermedian.transform(train[start_index:end_index,25:26])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,26:33])
	train[start_index:end_index,26:33] = imputerfrequency.transform(train[start_index:end_index,26:33])

	imputermedian = imputerformedian.fit(train[start_index:end_index,33:34])
	train[start_index:end_index,33:34] = imputermedian.transform(train[start_index:end_index,33:34])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,34:35])
	train[start_index:end_index,34:35] = imputerfrequency.transform(train[start_index:end_index,34:35])

	imputermedian = imputerformedian.fit(train[start_index:end_index,35:38])
	train[start_index:end_index,35:38] = imputermedian.transform(train[start_index:end_index,35:38])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,38:42])
	train[start_index:end_index,38:42] = imputerfrequency.transform(train[start_index:end_index,38:42])

	imputermedian = imputerformedian.fit(train[start_index:end_index,42:52])
	train[start_index:end_index,42:52] = imputermedian.transform(train[start_index:end_index,42:52])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,52:53])
	train[start_index:end_index,52:53] = imputerfrequency.transform(train[start_index:end_index,52:53])

	imputermedian = imputerformedian.fit(train[start_index:end_index,53:54])
	train[start_index:end_index,53:54] = imputermedian.transform(train[start_index:end_index,53:54])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,54:55])
	train[start_index:end_index,54:55] = imputerfrequency.transform(train[start_index:end_index,54:55])

	imputermedian = imputerformedian.fit(train[start_index:end_index,55:56])
	train[start_index:end_index,55:56] = imputermedian.transform(train[start_index:end_index,55:56])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,56:58])
	train[start_index:end_index,56:58] = imputerfrequency.transform(train[start_index:end_index,56:58])

	imputermedian = imputerformedian.fit(train[start_index:end_index,58:59])
	train[start_index:end_index,58:59] = imputermedian.transform(train[start_index:end_index,58:59])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,59:60])
	train[start_index:end_index,59:60] = imputerfrequency.transform(train[start_index:end_index,59:60])

	imputermedian = imputerformedian.fit(train[start_index:end_index,60:62])
	train[start_index:end_index,62:62] = imputermedian.transform(train[start_index:end_index,60:62])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,62:65])
	train[start_index:end_index,62:65] = imputerfrequency.transform(train[start_index:end_index,62:65])

	imputermedian = imputerformedian.fit(train[start_index:end_index,65:71])
	train[start_index:end_index,65:71] = imputermedian.transform(train[start_index:end_index,65:71])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,71:74])
	train[start_index:end_index,71:74] = imputerfrequency.transform(train[start_index:end_index,71:74])

	imputermedian = imputerformedian.fit(train[start_index:end_index,74:77])
	train[start_index:end_index,74:77] = imputermedian.transform(train[start_index:end_index,74:77])

	imputerfrequency = imputerforfrequency.fit(train[start_index:end_index,77:79])
	train[start_index:end_index,77:79] = imputerfrequency.transform(train[start_index:end_index,77:79])
	return train


def imputerStrategys(axis = 0):
	imputerformean = Imputer(missing_values = 'NaN', strategy = 'mean', axis = axis)
	imputerformedian = Imputer(missing_values = 'NaN', strategy = 'median', axis = axis)
	imputerforfrequency = Imputer(missing_values = 'NaN', strategy = 'most_frequent', axis = axis)
	return imputerformean,imputerformedian,imputerforfrequency

def dataframeToArray(df_train,df_test,df_real):	
	X_train = df_train.iloc[:,1:].values
	X_test = df_test.iloc[:,1:].values
	Y_test = df_real.iloc[:,1:].values
	X_test = np.hstack((X_test, Y_test))
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
	return dataset.columns.get_loc(col)
	
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