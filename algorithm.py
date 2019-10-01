# Algorithm   
from sklearn.metrics import mean_squared_error
from mlxtend.regressor import StackingCVRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
import lightgbm as lgb
from math import sqrt
import numpy as np
from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.kernel_ridge import KernelRidge
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import ElasticNet, Lasso
import xgboost as xgb

import warnings
def ignore_warn(*args, **kwargs):
    pass
warnings.warn = ignore_warn #ignore annoying warning (from sklearn and seaborn)


def Kernal(trainX, trainY, testX, testY, m1, m2, m3):
	
	w1 = 0.5
	w2 = 0.3
	w3 = 0.2

	TestResult = ((w1 * np.expm1(m1.predict(testX))) + \
      (w3 * np.expm1(m3.predict(testX))) + \
      (w2 * np.expm1(m2.predict(testX))))
	TestResult = np.around(TestResult, decimals = 2)	
	TrainResult = ((w1 * (m1.predict(trainX))) + \
      (w3 * (m3.predict(trainX))) + \
      (w2 * (m2.predict(trainX))))
	TrainResult = np.around(TrainResult, decimals = 2)
	print('RMSLE score on train data:', sqrt(mean_squared_error(trainY, TrainResult)))
	print('RMSLE score on test  data:', sqrt(mean_squared_error(testY, TestResult)))
	return TestResult

def rmsle_cv(model, train, y_train):
    kf = KFold(5, shuffle=True, random_state=42).get_n_splits(train.values)
    rmse= np.sqrt(-cross_val_score(model, train.values, y_train.values, scoring="neg_mean_squared_error", cv = kf))
    print("Mean of Model: ", np.around(rmse.mean(), decimals = 7), "Std of Model:", np.around(rmse.std(), decimals = 7))

# ridge, lasso, elasticnet, xgboost, lightgbm
def AlgoSCR(df_train, df_trainY, m1, m2, m3, m4, m5):
	model = StackingCVRegressor(regressors=(m1, m2, m3, m4, m5), meta_regressor=m1, use_features_in_secondary=True)	
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoSVR(df_train, df_trainY):
	model = SVR(kernel='rbf')
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoKRR(df_train, df_trainY):#
	model = KernelRidge(alpha=0.6, kernel='polynomial', degree=2, coef0=2.5)
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoGBR(df_train, df_trainY):
	model = GradientBoostingRegressor(n_estimators=3000, learning_rate=0.05,
                                   max_depth=4, max_features='sqrt',
                                   min_samples_leaf=15, min_samples_split=10, 
                                   loss='huber', random_state =5)
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoXgboost(df_train, df_trainY):#
	model = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoRForest(df_train, df_trainY):
	model = RandomForestRegressor(n_estimators = 300, random_state = 0)
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoLGBM(df_train, df_trainY):
	model = lgb.LGBMRegressor()
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoENet(df_train, df_trainY):#
	model = make_pipeline(RobustScaler(), ElasticNet(alpha=0.0005, l1_ratio=.9, random_state=3))
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

def AlgoLasso(df_train, df_trainY):#
	model = Lasso(alpha=0.00047)
	rmsle_cv(model, df_train, df_trainY)
	model.fit(df_train, df_trainY)
	result = model.predict(df_train)
	print("rms value of same set: ",np.around(sqrt(mean_squared_error(df_trainY, result)), decimals = 7))
	return model

#	model.fit(df_train, df_trainY)
#	pred = model.predict(df_train)
#	rms = sqrt(mean_squared_error(df_trainY, pred))
#	return rms