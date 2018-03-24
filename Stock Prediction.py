import numpy as np
import pandas as pd
import quandl
from matplotlib import pyplot as plt
from sklearn import preprocessing, cross_validation
from sklearn.linear_model import LinearRegression
import math
import pickle

df=quandl.get("NSE/DLINKINDIA")
df=df[['Close']]
#print (df)

forecast_col='Close'

forecast_out=int(math.ceil(0.001*len(df)))
#print(forecast_out)

df['Label']=df[forecast_col].shift(-forecast_out)

X=np.array(df.drop(['Label'],1))
X=preprocessing.scale(X)
X=X[:-forecast_out]
X_lately=X[-forecast_out:]
df.dropna(inplace=True)
y=np.array(df['Label'])

#print(len(X),len(y))

X_train,X_test,y_train,y_test=cross_validation.train_test_split(X,y,test_size=0.2)
clf=LinearRegression()
clf.fit(X_train,y_train)
with open('stock.pickle', 'wb') as f:
    pickle.dump(clf,f)

pickle_in=open('stock.pickle','rb')
clf=pickle.load(pickle_in)

accuracy=clf.score(X_test,y_test)

#print(accuracy)

forecast_set=clf.predict(X_lately)

print(forecast_set)