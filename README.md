# EXNO:4-DS
# AIM:
To read the given data and perform Feature Scaling and Feature Selection process and save the
data to a file.

# ALGORITHM:
STEP 1:Read the given Data.
STEP 2:Clean the Data Set using Data Cleaning Process.
STEP 3:Apply Feature Scaling for the feature in the data set.
STEP 4:Apply Feature Selection for the feature in the data set.
STEP 5:Save the data to the file.

# FEATURE SCALING:
1. Standard Scaler: It is also called Z-score normalization. It calculates the z-score of each value and replaces the value with the calculated Z-score. The features are then rescaled with x̄ =0 and σ=1
2. MinMaxScaler: It is also referred to as Normalization. The features are scaled between 0 and 1. Here, the mean value remains same as in Standardization, that is,0.
3. Maximum absolute scaling: Maximum absolute scaling scales the data to its maximum value; that is,it divides every observation by the maximum value of the variable.The result of the preceding transformation is a distribution in which the values vary approximately within the range of -1 to 1.
4. RobustScaler: RobustScaler transforms the feature vector by subtracting the median and then dividing by the interquartile range (75% value — 25% value).

# FEATURE SELECTION:
Feature selection is to find the best set of features that allows one to build useful models. Selecting the best features helps the model to perform well.
The feature selection techniques used are:
1.Filter Method
2.Wrapper Method
3.Embedded Method

# CODING AND OUTPUT:
```
import pandas as pd
import numpy as np
from scipy import stats
df=pd.read_csv("/content/bmi.csv")
df
```
```
df.head()
```
![Screenshot 2024-10-08 105722](https://github.com/user-attachments/assets/7487dd8f-0c0d-4bab-9abe-53fae9d7a3d0)
```
max_vals=np.max(np.abs(df[['Height','Weight']]))
max_vals
```
![Screenshot 2024-10-08 105746](https://github.com/user-attachments/assets/259d78dc-5449-4e53-8623-d29353cba0d3)
```
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
df[['Height','Weight']]=sc.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-10-08 105752](https://github.com/user-attachments/assets/4edaef8e-db63-4c45-9f8e-27abc54c5aa8)
```
from sklearn.preprocessing import MinMaxScaler
scaler=MinMaxScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head(10)
```
![Screenshot 2024-10-08 105903](https://github.com/user-attachments/assets/d19bd666-4cc9-43d3-bdba-801e378ead9c)
```
from sklearn.preprocessing import Normalizer
scaler=Normalizer()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-08 105935](https://github.com/user-attachments/assets/ab458e79-e391-4eaa-8ad2-aed7e8d91d6e)
```
from sklearn.preprocessing import MaxAbsScaler
scaler=MaxAbsScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df
```
![Screenshot 2024-10-08 110012](https://github.com/user-attachments/assets/f6750c8c-756b-4c92-acf0-3e381e7aa5cd)
```
from sklearn.preprocessing import RobustScaler
scaler=RobustScaler()
df[['Height','Weight']]=scaler.fit_transform(df[['Height','Weight']])
df.head()
```
![Screenshot 2024-10-08 110039](https://github.com/user-attachments/assets/092b1d55-bb29-4400-bc12-b9b6b0634558)
```
import pandas as pd
import numpy as np
import seaborn as sns
from scipy.stats import chi2_contingency
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
df =pd.read_csv('/content/income(1) (1).csv',na_values=[" ?"])
df
```
![Screenshot 2024-10-08 110141](https://github.com/user-attachments/assets/c89e9b38-8843-45f0-a455-b7653eecbc0f)
```
missing=df[df.isnull().any(axis=1)]
missing
```
![Screenshot 2024-10-08 110220](https://github.com/user-attachments/assets/0357bf36-c61c-4816-ab55-c01d5a2e515c)
```
df.isnull().sum()
```
![Screenshot 2024-10-08 110251](https://github.com/user-attachments/assets/ed214ff9-a9f7-4d5d-a7ad-44947c7c01bb)
```
df1=df.dropna(axis=0)
df1
```
![Screenshot 2024-10-08 110328](https://github.com/user-attachments/assets/b89fe528-d0ba-45de-b55c-ef0ce1d25c06)
```
sal=df['SalStat']
df1['SalStat']=df1['SalStat'].map({' less than or equal to 50,000':0,' greater than 50,000':1})
print(df1['SalStat'])
```
![Screenshot 2024-10-08 110437](https://github.com/user-attachments/assets/5d6de6f7-1267-457d-bc64-a64f09edbefd)
```
sal1=df['SalStat']
dfs=pd.concat([sal,sal1],axis=1)
dfs
```
![Screenshot 2024-10-08 110525](https://github.com/user-attachments/assets/a9cf041d-d884-4e37-929a-c324ee89c172)
```
new_data=pd.get_dummies(df1,drop_first=True)
new_data
```
![image](https://github.com/user-attachments/assets/01e6facb-d8ba-4e15-b70d-0a0fc25d39a6)
```
columns_list=list(new_data.columns)
print(columns_list)
```
![Screenshot 2024-10-08 110626](https://github.com/user-attachments/assets/b9c76fca-b76b-456f-9bd5-fc396d38c6d0)
```
features=list(set(columns_list)-set(['SalStat']))
print(features)
```
![Screenshot 2024-10-08 110643](https://github.com/user-attachments/assets/f53cd9f6-e65a-481c-abfa-599c3a54be6c)
```
y=new_data['SalStat'].values
print(y)
```
![Screenshot 2024-10-08 110809](https://github.com/user-attachments/assets/e5aa5e67-78c2-4bf8-8c35-1f0097d27837)
```
x=new_data[features].values
print(x)
```
![Screenshot 2024-10-08 110835](https://github.com/user-attachments/assets/feb4970f-c87a-4e42-a34f-10ed22684324)
```
train_x,test_x,train_y,test_y=train_test_split(x,y,test_size=0.3, random_state=0)
KNN_classifier=KNeighborsClassifier(n_neighbors=5)
KNN_classifier.fit(train_x, train_y)
```
![Screenshot 2024-10-08 110924](https://github.com/user-attachments/assets/71f9da49-32df-4def-8726-94a919547dfe)
```
prediction=KNN_classifier.predict(test_x)
confusionMatrix=confusion_matrix(test_y,prediction)
print(confusionMatrix)
```
![Screenshot 2024-10-08 110957](https://github.com/user-attachments/assets/a43dc458-94b7-4c41-8581-e1bb5d4c9f49)
```
accuracy_score=accuracy_score(test_y,prediction)
print(accuracy_score)
```
![Screenshot 2024-10-08 111007](https://github.com/user-attachments/assets/a4e0f20e-5c81-45ed-a731-ea61e59fa0ff)
```
print('Misclassified samples: %d' %(test_y!=prediction).sum())
```
![Screenshot 2024-10-08 111013](https://github.com/user-attachments/assets/ec3621be-962a-48ad-8cc7-5743df824c45)
```
df.shape
```
![Screenshot 2024-10-08 111018](https://github.com/user-attachments/assets/b0665e68-9045-44cf-b0cb-565bcea95c65)
```
  tips=sns.load_dataset('tips')
  tips.head()
```
![Screenshot 2024-10-08 111149](https://github.com/user-attachments/assets/bae2e9d1-8e2e-425a-9e63-13ad857ea6fd)
```
contingency_table=pd.crosstab(tips['sex'],tips['time'])
print(contingency_table)
```
![Screenshot 2024-10-08 111156](https://github.com/user-attachments/assets/5a6bc592-5d9f-47d4-84cb-54bae0127bd5)
```
chi2,p, _, _=chi2_contingency(contingency_table)
print(f"Chi-Square Statistic: {chi2}")
print(f"p-vale: {p}")
```
![Screenshot 2024-10-08 111201](https://github.com/user-attachments/assets/d7513430-0c4a-4862-9f78-2202777f8cd7)
```
import pandas as pd
from sklearn.feature_selection import SelectKBest,mutual_info_classif,f_classif
data={
    'Feature1':[1,2,3,4,5],
    'Feature2':['A','B','C','A','B'],
    'Feature3':[0,1,1,0,1],
    'Target':[0,1,1,0,1]
}
df=pd.DataFrame(data)
df
```
![Screenshot 2024-10-08 111342](https://github.com/user-attachments/assets/f8a62a3f-9771-4f8b-8049-899c06d1a7d8)
```
X=df[['Feature1','Feature3']]
y=df['Target']
selector=SelectKBest(score_func=mutual_info_classif,k=1)
X_new=selector.fit_transform(X,y)
selected_features_indices=selector.get_support(indices=True)
selected_features=X.columns[selected_features_indices]
print("Selected Features:")
print(selected_features)
```
![Screenshot 2024-10-08 111425](https://github.com/user-attachments/assets/72333a6e-e94d-4eaf-8b9b-e23ee148692b)

# RESULT:
       Thus the code for feature scaling and encoding executed successfully.
