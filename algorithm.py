
try:
	from functions import dataframeToArray, getColoumIndex, missingValueHandler, getDatasets, correlationMatrix
except:
    print("HeaderFiles not in access")
    raise SystemExit    

# Import dataset
df_train,df_test,df_real = getDatasets()

# Analysis
correlationMatrix(df_train, "SalePrice", False, 3)

# OverallQual = 0.79, relate most
index = getColoumIndex(df_train, "OverallQual")

# Dataframe To Arrays and remove Id's
X_train, X_test = dataframeToArray(df_train,df_test,df_real)

# Sort on the base of coloum values OverallQual
X_train,X_test = missingValueHandler(X_train,X_test,index)

