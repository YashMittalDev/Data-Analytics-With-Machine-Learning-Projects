import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#EDA(Exploratry Data Analysis)

#data Reading from a csv file
#to work with Dates using parse_Dates=True
avacado = pd.read_csv(r"E:\data science lectures\2nd resume project\2nd- reg resume project\RESUME PROJECT -- PRICE PREDICTION\avocado.csv").reset_index(drop=True)
avacado.head()

#step 1:- Removing extra unnamed column from dataframe

avacado=avacado.drop('Unnamed: 0',axis=1)

#step 2:- check dataset: 

avacado.info()

#step 3 :- change object type columns into category columns  

avacado.type=avacado['type'].astype('category')
avacado.region = avacado.region.astype('category')

#step 4 :- Data Cleaning:-

#a)  to check null values in dataset:-

avacado.isnull().sum()

#no null values are here.....

#b) to check if the dataset have any unknown values

avacado.type.unique()
avacado.region.unique()

#so no unsound values
 
#step 5:- now find Correlation between datasets:- 

corr = avacado.corr()

#step 6 :- preparing Pivot Table:- 

pivot_table = avacado.pivot_table(index=['year','type'] , aggfunc = [min,max,np.mean,np.median] , values= 'AveragePrice')

#step 7:- visualizing the data:- 

#histograms :- 

#about the information about the price

plt.hist(avacado.AveragePrice)
#mostly the average price is in between 1-2

plt.bar(avacado.region.unique(),height = avacado.groupby('region')["AveragePrice"].mean())

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------

#----Predicting the price of Avacados----#

import warnings 
warnings.

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


avacado = pd.read_csv(r"E:\data science lectures\2nd resume project\2nd- reg resume project\RESUME PROJECT -- PRICE PREDICTION\avocado.csv").reset_index(drop=True)
avacado.head()
avacado.info()
#
sns.distplot(avacado['AveragePrice'])

#
sns.countplot(data=avacado , x = 'year' , hue='type')
#insight:- sales of 2015,2016,2017,2018(lesser sales)

#
sns.boxplot(x='type',y='AveragePrice',data=avacado)
#insight ---> conventional price is little less than organic price

#
sns.boxenplot(x='year',y='AveragePrice',data=avacado)
#--> price of avacado is little higher in 2017 (there were shortage due to some reason)

#Dealing with categorical features
avacado.type = avacado.type.map({'conventional':0,'organic':1})
#extracting date,month from date column:-
avacado.Date = avacado.Date.apply(pd.to_datetime)
avacado['Month']=avacado.Date.apply(lambda x:x.month)
avacado.drop('Date',axis=1,inplace=True)
avacado.Month=avacado.Month.map({1:'Jan',2:'Feb',3:'MARCH',4:'April',5:'May',6:'June',7:'July',8:'Aug',9:'Sept',10:'Oct',11:'Nov',12:'Dec'})

#insight on monthly sales
sns.countplot(x=avacado.Month)
plt.title("Monthly Sales Distribution",fontdict={'fontsize':25})

#highest sales - January , lowest sales in :- June , so sales rise in feb march jan.


#------------Model Creation----------------

#1. Create categorical data into numerical data :- 1) dummy variable 2) one hot encoder 3) label encoder

dummies = pd.get_dummies(avacado[['year','region','Month']],drop_first=True)
df_dummies = pd.concat([avacado.iloc[:,2:10],dummies],axis=1)
target = avacado.AveragePrice
# df_dummies = indepenedent variable, target=dependent Variable

#2. Splitting the dataset ----

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(df_dummies,target,test_size=0.30)

#3:- stardize the data:-
from sklearn.preprocessing import StandardScaler
SS = StandardScaler()
x_train = SS.fit_transform(x_train)
x_test = SS.transform(x_test) 

#4 :- Importing models:- 

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score,accuracy_score


regressor_algos = {
    "Linear Regression":LinearRegression(),
    "Decision Tree ":DecisionTreeRegressor(),
    "Random Forest Regression":RandomForestRegressor(),
    "Support Vector":SVR(),
    "K neighbors Regression":KNeighborsRegressor(n_neighbors=1),
    "XGBOOST":XGBRegressor()
    }

Results = pd.DataFrame(columns=['MSE','MAE','R2 Score'])

for method,func in regressor_algos.items():
    model = func.fit(x_train,y_train)
    pred = model.predict(x_test)
    Results.loc[method]=[np.round(mean_squared_error(y_test,pred),3),
                        np.round(mean_absolute_error(y_test,pred),3),
                        np.round(r2_score(y_test,pred),3),
                        ]
    

target_mean = np.round(0.1 * target.mean(),3)

Results.sort_values('R2 Score',ascending=False).style.background_gradient(cmap='Greens',subset=['R2 Score'])

#so Random Forest  , XgBoost ,  Knn is best performing model.












 

 