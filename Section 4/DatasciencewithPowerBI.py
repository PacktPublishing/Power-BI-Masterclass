import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.model_selection import train_test_split
import numpy as np
from xgboost import XGBClassifier

X = dataset.select_dtypes(include=['number'])
y = X.pop('Churn')

X_train,X_test,y_train,y_test = train_test_split(X, y, random_state=42,stratify=y)

#Scale data
scaler = MinMaxScaler()
scaler.fit(X_train,y_train)
X_train_scaled = scaler.transform(X_train)
X_test_scaled = scaler.transform(X_test)
xgb = XGBClassifier()
xgb.fit(X_train_scaled,y_train)
prediction = xgb.predict(X_test_scaled)
prediction = xgb.predict(X_test_scaled)
results = X_test.reset_index().join(pd.DataFrame(prediction, columns=['prediction']))
final = pd.merge(results,dataset['Company'], left_on='index', right_index=True)
final.head()
