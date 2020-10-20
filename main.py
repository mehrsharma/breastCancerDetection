#Breast cancer detection based on wisconsin data, extracted from https://www.kaggle.com/uciml/breast-cancer-wisconsin-data
#Refer to https://medium.com/@randerson112358/breast-cancer-detection-using-machine-learning-38820fe98982
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

df = pd.read_csv('data.csv')

#drop the column with missing values (unnamed 32)
df = df.dropna(axis=1) #indicates its a column and not a row

#get the new number of rows, columns
#print(df.shape)

#encode the categorical data values
labelencoder_Y = LabelEncoder()
df.iloc[:,1] = labelencoder_Y.fit_transform(df.iloc[:,1].values)

#get the correlation of the columns
#print(df.iloc[:,1:12].corr())

#visualizing the correlation
plt.figure(figsize = (10,10))
sns.heatmap(df.iloc[:,1:12].corr(), annot=True, fmt = '.0%')
plt.show()


#training


#split the data into independent(X) and dependent(Y) data sets
X = df.iloc[:,2:31].values
Y = df.iloc[:,1].values

#split the data into 1/4 testing, 3/4 training
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25 , random_state = 0)

#feature scaling
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#creating a function for the models
def models(X_train, Y_train):
    #logistic regression
    from sklearn.linear_model import LogisticRegression
    log = LogisticRegression(random_state = 0)
    log.fit(X_train, Y_train)
    #decision tree
    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
    tree.fit(X_train, Y_train)
    #random forest classifier
    from sklearn.ensemble import RandomForestClassifier
    forest = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
    forest.fit(X_train, Y_train)
    #print the models' accuracy on the training data
    print('[0]Logistic Regression Training Accuracy:', log.score(X_train, Y_train))
    print('[1]Decision Tree Training Accuracy:', tree.score(X_train, Y_train))
    print('[2]Random Forest Training Accuracy:', forest.score(X_train, Y_train))
    return log,tree,forest

#getting all of the models
model = models(X_train, Y_train)
print(model)

#testing

for i in range (len(model)):
    print("Model ", i )
    cm = confusion_matrix(Y_test, model[i].predict(X_test))
    TP = cm[0][0]
    TN = cm[1][1]
    FN = cm[1][0]
    FP = cm[0][1]

    print(cm)
    print('Testing Accuracy = ', (TP + TN)/(TP + TN + FN + FP))
    print()
