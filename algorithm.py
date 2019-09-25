# Algorithm   
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from math import sqrt

def AlgoXgboost(df_train, df_trainY, df_test, df_testY):
	model_xgb = xgb.XGBRegressor(colsample_bytree=0.4603, gamma=0.0468, 
                             learning_rate=0.05, max_depth=3, 
                             min_child_weight=1.7817, n_estimators=2200,
                             reg_alpha=0.4640, reg_lambda=0.8571,
                             subsample=0.5213, silent=1,
                             random_state =7, nthread = -1)

	model_xgb.fit(df_train, df_trainY)
	xgb_pred = model_xgb.predict(df_test)
	rms = sqrt(mean_squared_error(df_testY, xgb_pred))
	return rms