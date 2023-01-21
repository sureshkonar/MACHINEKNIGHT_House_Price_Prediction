import pandas as pd
#importing file 
data = pd.read_csv('train.csv')
#printing values and analysis of data
for column in data.columns:
    print(data[column].value_counts())
    print("*"*20)
    data.describe()

#data cleaning & Feature Engineering
data.drop(columns=['gym','id','lift','latitude','longitude','activation_date','property_age','water_supply','building_type','swimming_pool','negotiable','bathroom','furnishing', 'parking','facing','cup_board','floor','total_floor','amenities','amenities','balconies','lease_type'], inplace=True)
print(data['locality'].value_counts())
print(data['type'].value_counts())
print(data.describe())
print(data['property_size'].value_counts())
print(data['property_size'].unique())
data['locality']= data['locality'].apply(lambda x:x.strip())
locality_count= data['locality'].value_counts()
locality_count_less_10 = locality_count[locality_count<=10]
print(locality_count_less_10)
data['locality']=data['locality'].apply(lambda x: 'other' if x in locality_count_less_10 else x ) 
print(data['locality'].value_counts())
data=data[(data['property_size']>=600)]
data['rent'].astype(int)
print(data.describe())
print(data.shape)
data['bhk'] = data['type'].str.get(-1)
data.to_csv('cleandata.csv')


#importing all modules required for prediction model
X=data.drop(columns=['rent'])
y=data['property_size']
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression,Lasso,Ridge
from sklearn.preprocessing import OneHotEncoder,StandardScaler
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score

#setting up the initial values
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
print(X_train.shape)
print(X_train)
print(X_test.shape)
print(y_train)

# #Applying Linear Regresssion
column_trans = make_column_transformer((OneHotEncoder(sparse=False),['locality']),remainder='passthrough')
scaler=StandardScaler()
lr=LinearRegression(normalize=True)
pipe=make_pipeline(column_trans,scaler,lr)
pipe.fit(X_train,y_train)
y_pred_lr=pipe.predict(X_test)
r2_score(y_test,y_pred_lr)

# Applying Lasso
lasso=Lasso()
pipe=make_pipeline(column_trans,scaler,lasso)
# pipe.fit(X_train,y_train)
y_pred_lasso =pipe.predict(X_test)
r2_score(y_test,y_pred_lasso)

#Applying Ridge
ridge=Ridge()
pipe=make_pipeline(column_trans,scaler,ridge)
pipe.fit(X_train,y_train)
y_pred_ridge = pipe.predict(X_test)
r2_score(y_test,y_pred_ridge)

#extracting the generated data in pkl format
import pickle
pickle.dump( open('RidgeModel.pkl','wb'))
