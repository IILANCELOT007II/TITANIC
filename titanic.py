###############################
#LANCELOTXV
###############################



import pandas as pd
import numpy as np
import random as rnd

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')
dataset =[train_df,test_df]

desc = train_df.describe()

#since name titles help in grouping age groups extracting titles from name and discarding name feature itself
for data in dataset:
    data['Title'] = data.Name.str.extract(' ([A-Za-z]+)\.', expand=False)

pd.crosstab(train_df['Title'], train_df['Sex'])

for data in dataset:
    data['Title'] = data['Title'].replace(['Lady', 'Countess','Capt', 'Col',\
 	'Don', 'Dr', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')

    data['Title'] = data['Title'].replace('Mlle', 'Miss')
    data['Title'] = data['Title'].replace('Ms', 'Miss')
    data['Title'] = data['Title'].replace('Mme', 'Mrs')
    
train_df[['Title', 'Survived']].groupby(['Title'], as_index=False).mean()

title_mapping = {"Mr": 0, "Miss": 1, "Mrs": 2, "Master": 3, "Rare": 4}
for data in dataset:
    data['Title'] = data['Title'].map(title_mapping)
    data['Title'] = data['Title'].fillna(0)

#Dropping cabin, ticket, name, passengerid features as dont contribute to prediction
train_df = train_df.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)
test_df = test_df.drop(['Ticket', 'Cabin','Name', 'PassengerId'], axis=1)
dataset = [train_df, test_df]

for data in dataset:
    data['Sex'] = data['Sex'].map( {'female': 1, 'male': 0} ).astype(int)
    

from sklearn.impute import SimpleImputer
imputer = SimpleImputer()
data = train_df.iloc[:, 3].values
data = data.reshape(-1,1)
train_df['Age'] =imputer.fit_transform(data)
train_df['Age'].astype(int)
data = test_df.iloc[:, 2].values
data = data.reshape(-1,1)
test_df['Age'] =imputer.fit_transform(data)
test_df['Age'].astype(int)

dataset = [train_df, test_df]


train_df['AgeGroup'] = pd.cut(train_df['Age'], 5)
train_df[['AgeGroup', 'Survived']].groupby(['AgeGroup'], as_index=False).mean().sort_values(by='AgeGroup', ascending=True)
#Kids had the highest survial rate i.e age=0-16
#Creating age groups based on this

for data in dataset:    
    data.loc[ data['Age'] <= 16, 'Age'] = 0
    data.loc[(data['Age'] > 16) & (data['Age'] <= 32), 'Age'] = 1
    data.loc[(data['Age'] > 32) & (data['Age'] <= 48), 'Age'] = 2
    data.loc[(data['Age'] > 48) & (data['Age'] <= 64), 'Age'] = 3
    data.loc[ data['Age'] > 64, 'Age'] = 4
train_df = train_df.drop(['AgeGroup'], axis=1)
dataset = [train_df, test_df]

#parch and sibsp have high correlation therefore combing them into a single feature

for data in dataset:
    data['FamilySize'] = data['SibSp'] + data['Parch'] + 1

train_df[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False)
#family with size = 4 had the highest survival rate


for data in dataset:
    data['Family'] = 0
    data.loc[data['FamilySize'] == 1, 'Family'] = 1

train_df[['Family', 'Survived']].groupby(['Family'], as_index=False).mean()

train_df = train_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
test_df = test_df.drop(['Parch', 'SibSp', 'FamilySize'], axis=1)
dataset = [train_df, test_df]

freq_port = train_df.Embarked.dropna().mode()[0]
for data in dataset:
    data['Embarked'] = data['Embarked'].fillna(freq_port)
    
train_df[['Embarked', 'Survived']].groupby(['Embarked'], as_index=False).mean().sort_values(by='Survived', ascending=False)

for data in dataset:
    data['Embarked'] = data['Embarked'].map( {'S': 0, 'C': 1, 'Q': 2} ).astype(int)

test_df['Fare'].fillna(test_df['Fare'].dropna().median(), inplace=True)

data = test_df.iloc[:, 3].values
data = data.reshape(-1,1)
test_df['Fare'] =imputer.fit_transform(data)
test_df['Fare'].astype(int)

train_df['FareBand'] = pd.qcut(train_df['Fare'], 4)
train_df[['FareBand', 'Survived']].groupby(['FareBand'], as_index=False).mean().sort_values(by='FareBand', ascending=True)

for data in dataset:
    data.loc[ data['Fare'] <= 7.91, 'Fare'] = 0
    data.loc[(data['Fare'] > 7.91) & (data['Fare'] <= 14.454), 'Fare'] = 1
    data.loc[(data['Fare'] > 14.454) & (data['Fare'] <= 31), 'Fare']   = 2
    data.loc[ data['Fare'] > 31, 'Fare'] = 3
    data['Fare'] = data['Fare'].astype(int)

train_df = train_df.drop(['FareBand'], axis=1)
dataset = [train_df, test_df]

X = train_df.iloc[:, 1:8].values
y = train_df.iloc[:, 0].values


from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer


onehotencoder = OneHotEncoder(categories=[3], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

onehotencoder = OneHotEncoder(categories=[5], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

onehotencoder = OneHotEncoder(categories=[4], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [7])],remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

onehotencoder = OneHotEncoder(categories=[3], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [10])],remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

onehotencoder = OneHotEncoder(categories=[5], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [12])],remainder='passthrough')
X = ct.fit_transform(X)
X = X[:, 1:]

X_test = test_df.iloc[:, :].values


onehotencoder = OneHotEncoder(categories=[3], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [0])],remainder='passthrough')
X_test = ct.fit_transform(X_test)
X_test = X_test[:, 1:]

onehotencoder = OneHotEncoder(categories=[5], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [3])],remainder='passthrough')
X_test = ct.fit_transform(X_test)
X_test = X_test[:, 1:]

onehotencoder = OneHotEncoder(categories=[4], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [7])],remainder='passthrough')
X_test = ct.fit_transform(X_test)
X_test = X_test[:, 1:]

onehotencoder = OneHotEncoder(categories=[3], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [10])],remainder='passthrough')
X_test = ct.fit_transform(X_test)
X_test = X_test[:, 1:]

onehotencoder = OneHotEncoder(categories=[5], dtype=np.float64)   
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [12])],remainder='passthrough')
X_test = ct.fit_transform(X_test)
X_test = X_test[:, 1:]



from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)



# Fitting classifier to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel='rbf')
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)


from sklearn.metrics import accuracy_score
score = accuracy_score(y_test, y_pred)
print('Accuracy score: %f' % score)






