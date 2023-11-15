# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
 NAME : mohammed imthiyas.M
 REG NO : 212222230083
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
data=pd.read_csv("titanic_dataset.csv")
data.head()
data.isnull().sum()
sns.heatmap(data.isnull(),cbar=False)
plt.title("sns.heatmap(data.isnull(),cbar=False)")
plt.show()
data['Age'] = data['Age'].fillna(data['Age'].dropna().median())
data['Embarked']=data['Embarked'].fillna('S')
data.loc[data['Sex']=='male','Sex']=0
data.loc[data['Sex']=='female','Sex']=1
data.loc[data['Embarked']=='S','Embarked']=0
data.loc[data['Embarked']=='C','Embarked']=1
data.loc[data['Embarked']=='Q','Embarked']=2
drop_elements = ['Name','Cabin','Ticket']
data = data.drop(drop_elements, axis=1)
sns.heatmap(data.corr(),annot=True,fmt= '.1f',ax=ax)
plt.title("HeatMap")
plt.show()
sns.heatmap(data.isnull(),cbar=False)            
plt.show()
data.Survived.value_counts(normalize=True).plot(kind='bar',alpha=0.5)                  
plt.show()
data.Pclass.value_counts(normalize=True).plot(kind='bar',alpha=0.5)
plt.show()
plt.scatter(data.Survived,data.Age,alpha=0.1)
plt.title("Age with Survived")                                
plt.show()
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
X = data.drop("Survived",axis=1)
y = data["Survived"]
mdlsel = SelectKBest(chi2, k=5)
mdlsel.fit(X,y)
ix = mdlsel.get_support()
data2 = pd.DataFrame(mdlsel.transform(X), columns = X.columns.values[ix])
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
target = data['Survived'].values
data_features_names = ['Pclass','Sex','SibSp','Parch','Fare','Embarked','Age']
features = data[data_features_names].values
#Build test and training test
X_train,X_test,y_train,y_test = train_test_split
      (features,target,test_size=0.3,random_state=42)
my_forest=RandomForestClassifier(max_depth=5,min_samples_split=10,
                n_estimators=500,random_state=5,criterion='entropy')
my_forest_ = my_forest.fit(X_train,y_train)
target_predict=my_forest_.predict(X_test)
print("Random forest score: ",accuracy_score(y_test,target_predict))
from sklearn.metrics import mean_squared_error, r2_score
print ("MSE    :",mean_squared_error(y_test,target_predict))
print ("R2     :",r2_score(y_test,target_predict))
```

# OUPUT


![283035791-738ce517-e2e4-44f6-bf62-23e3c5c98653](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/2239d6ff-4eb5-48a7-aa6e-0e2d39f1d957)








![283035800-e53f526a-3f49-4c88-a46f-900183605b16](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/501163be-0aa7-4b32-9fe2-f0d14968cc9a)






![283035806-ff9d35a6-2777-467c-8eea-a34db19eddbc](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/988f6c70-d09e-483a-a44b-2d1b6b6a7da0)






![283035823-9e92eddf-f178-4b08-8c68-5c193e738063](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/b34de629-c324-4dc2-93c4-ba642e20fc9f)






![283035823-9e92eddf-f178-4b08-8c68-5c193e738063](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/31fc47b3-7efd-4643-b59a-2b15d9c21d96)






![283035828-cb2e2913-1aa0-4ed6-8001-45e981851727](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/54880320-0c89-4584-851d-8565fc73e123)


![283035840-83c22975-ec11-4bd7-a376-d29c3388f7f1](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/7cd1be1c-fbc8-4dbf-9a36-24c6e00963a6)





![283035843-9716bc34-bb3f-4da2-9953-a86dd8028df4](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/0dc50e7e-0f7a-415a-a268-9f345e01d6c5)




![283035852-6d798334-a665-4ae2-a5fe-584d6df45f31](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/75f69204-ee0c-4e78-bedb-44c50213f955)

![283035868-0ab03c4c-89d1-417a-8ba7-1adc7f5443a8](https://github.com/imthiyas19/ODD2023-Datascience-Ex-07/assets/120353416/7f74fd37-8d5e-4c71-ab4d-1fa4a8132fc1)




