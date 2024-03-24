import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
plt.style.use('fivethirtyeight')

data = pd.read_csv('testset.csv')
print(data.columns)
print(data.shape)
print(data.isnull().sum())
data.columns = data.columns.str.strip() #to remove trailing whitespaces.
a = data.drop(['_precipm','_windchillm'],axis=1,inplace = True)

print(data.columns)
print(data.info())

#Data_Analysis:-
data['datetime_utc'] = pd.to_datetime(data['datetime_utc'])
data['month']= pd.to_datetime(data['datetime_utc']).dt.month
data['year']= pd.to_datetime(data['datetime_utc']).dt.year
data['day']= data['datetime_utc'].dt.day
data['hour']=data['datetime_utc'].dt.hour
data['Weekday']= data['datetime_utc'].dt.weekday
print(data.head(10))
data.drop(['datetime_utc'],axis=1 , inplace= True )
value_count = data['_conds'].value_counts() #we will use one hot encoding method for it.
print(value_count)
value_count2 = data['_wdire'].value_counts()
x1 = pd.crosstab([data._rain,data._conds],[data.month],margins = True)
print(x1)
print(value_count2)  #we need to use one hot encoding for it also.
data.dropna(subset=['_conds','_wdire'],inplace = True)
#Analysis and prediction modelling:-

from sklearn.preprocessing import OneHotEncoder
column_name = '_conds'
column_to_encode = data[[column_name]]
column_to_encode_2d = column_to_encode.values.reshape(-1, 1)
encoder = OneHotEncoder(sparse_output=False, drop='first')  # Setting drop='first' to avoid dummy variable trap
encoded_data = encoder.fit_transform(column_to_encode_2d)
encoded_df = pd.DataFrame(encoded_data, columns=encoder.get_feature_names_out([column_name]))
data = pd.concat([data, encoded_df], axis=1)
data.drop([column_name], axis=1, inplace=True)
column_name2 = '_wdire'
column_to_encode2= data[[column_name2]]
column_to_encode2_2d = column_to_encode2.values.reshape(-1,1)
encoder2 = OneHotEncoder(sparse_output=False,drop= 'first')
encoded_data2= encoder.fit_transform(column_to_encode2_2d)
encoded_df2 = pd.DataFrame(encoded_data2, columns=encoder.get_feature_names_out([column_name2]))
data = pd.concat([data, encoded_df2], axis=1)
data.drop([column_name2], axis=1, inplace=True)
data = data.dropna(subset=['_rain'])
print(data.head())
print(data)
print(data.head())
print(data._rain.isnull().sum())
included_cols = ['_dewptm', '_fog', '_hail','_heatindexm', '_hum', '_pressurem', '_rain', '_snow','_tempm', '_thunder', '_tornado', '_vism', '_wdird','_wgustm', '_wspdm']
# plt1 = sns.heatmap(data[included_cols].corr(),annot = True, annot_kws={'size':10},cmap='RdYlGn',linewidths =0.2 )
# plt.show()
#from the co-relation graph we got to know that thunder,humidity are some important feature to predict whether it will rain or not.
print(data.info())
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='mean')
data = pd.DataFrame(imputer.fit_transform(data),columns=data.columns)
# plt1 = sns.heatmap(data[included_cols].corr(),annot = True, annot_kws={'size':10},cmap='RdYlGn',linewidths =0.2 )
# plt.show()
print(data.info())
from sklearn.model_selection import train_test_split
train_df,val_df= train_test_split(data,random_state = 42, test_size=0.25)
print(train_df.columns)
input_cols = ['_dewptm', '_fog', '_hail', '_heatindexm', '_hum', '_pressurem',
        '_snow', '_tempm', '_thunder', '_tornado', '_vism', '_wdird',
       '_wgustm', '_wspdm', 'month', 'day', 'year', 'hour', 'Weekday',
       '_conds_Clear', '_conds_Drizzle', '_conds_Fog', '_conds_Funnel Cloud',
       '_conds_Haze', '_conds_Heavy Fog', '_conds_Heavy Rain',
       '_conds_Heavy Thunderstorms and Rain',
       '_conds_Heavy Thunderstorms with Hail', '_conds_Light Drizzle',
       '_conds_Light Fog', '_conds_Light Freezing Rain',
       '_conds_Light Hail Showers', '_conds_Light Haze', '_conds_Light Rain',
       '_conds_Light Rain Showers', '_conds_Light Sandstorm',
       '_conds_Light Thunderstorm', '_conds_Light Thunderstorms and Rain',
       '_conds_Mist', '_conds_Mostly Cloudy', '_conds_Overcast',
       '_conds_Partial Fog', '_conds_Partly Cloudy', '_conds_Patches of Fog',
       '_conds_Rain', '_conds_Rain Showers', '_conds_Sandstorm',
       '_conds_Scattered Clouds', '_conds_Shallow Fog', '_conds_Smoke',
       '_conds_Squalls', '_conds_Thunderstorm',
       '_conds_Thunderstorms and Rain', '_conds_Thunderstorms with Hail',
       '_conds_Unknown', '_conds_Volcanic Ash', '_conds_Widespread Dust',
       '_wdire_ESE', '_wdire_East', '_wdire_NE', '_wdire_NNE', '_wdire_NNW',
       '_wdire_NW', '_wdire_North', '_wdire_SE', '_wdire_SSE', '_wdire_SSW',
       '_wdire_SW', '_wdire_South', '_wdire_Variable', '_wdire_WNW',
       '_wdire_WSW', '_wdire_West', '_wdire_nan']
target_cols= ['_rain']
train_inputs = train_df[input_cols]
train_targets = train_df[target_cols]
val_inputs = val_df[input_cols]
val_targets = val_df[target_cols]


from sklearn.metrics import accuracy_score




#Logistic Regression model :-
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
model1 = LogisticRegression()
model1.fit(train_inputs,train_targets)
pred1 = model1.predict(train_inputs)
pred2 = model1.predict(val_inputs)
a1 = accuracy_score(pred1,train_targets)
a2 = accuracy_score(pred2,val_targets)
print(a1)
print(a2)

#Random Forest Classifier :-
from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier()
model2.fit(train_inputs,train_targets)
pred3 = model2.predict(train_inputs)
pred4 = model2.predict(val_inputs)
a3 = accuracy_score(pred3,train_targets)
a4 = accuracy_score(pred4,val_targets)
print(a3)
print(a4)

#SGD classifier :-
from sklearn.linear_model import SGDClassifier
model3 = SGDClassifier()
model3.fit(train_inputs,train_targets)
pred5 = model3.predict(train_inputs)
pred6 = model3.predict(val_inputs)
a5 = accuracy_score(pred3,train_targets)
a6 = accuracy_score(pred4,val_targets)
print(a5)
print(a6)
from sklearn.metrics import confusion_matrix
y1 = confusion_matrix(val_targets,pred2,normalize = 'true')
sns.heatmap(y1,annot= True,annot_kws={'size':20},linewidths=0.2,cmap='RdYlGn')
plt.show()  #our model has a very good accuracy.
print(train_inputs.shape)






