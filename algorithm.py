from sklearn.linear_model import LinearRegression

def linearRegression(X_train, Y_train, X_test):
	obj = LinearRegression()
	obj.fit(X_train, Y_train)
	Y_pred = obj.predict(X_test)
	return Y_pred