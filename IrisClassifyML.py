#for reading data
import pandas as pd

#to make output look pretty rather than gunked up with warnings
import warnings
warnings.simplefilter(action="ignore")

#for visualization
import matplotlib.pyplot as plt
import seaborn as sns

#for building and analyzing different models
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

#load data
filename = 'iris.data'
data = pd.read_csv(filename)

#plotting data
sns.pairplot(data,hue="classification")
plt.show()

sns.swarmplot(x='classification', y='sepal_length', data=data, hue="classification")
plt.show()
sns.swarmplot(x='classification', y='sepal_width', data=data, hue="classification")
plt.show()
sns.swarmplot(x='classification', y='petal_length', data=data, hue="classification")
plt.show()
sns.swarmplot(x='classification', y='petal_width', data=data, hue="classification")
plt.show()

#split data up into test and training sets
ptrain, ptest = train_test_split(data, test_size = 0.5) #creates two data sets, where ptrain is .5 of the data points and ptest is (1-.5) of the data points
xTrain= ptrain[['sepal_length','sepal_width','petal_length','petal_width']]
yTrain= ptrain.classification
xTest= ptest[['sepal_length','sepal_width','petal_length','petal_width']]
yTest= ptest.classification

#logistic regression model test
from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)
# Summary of the predictions made by the classifier
print(classification_report(yTest, yPred))
# Accuracy score
print('The accuracy of the Logistic Regression is' + str(accuracy_score(yTest, yPred)))

#K-Nearest Neighbors test
#2-KNN
from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=2)
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)
# Summary of the predictions made by the classifier
print(classification_report(yTest, yPred))
# Accuracy score
print('The accuracy of 2-KNN is: ' + str(accuracy_score(yTest, yPred)))

#4-KNN
model = KNeighborsClassifier(n_neighbors=4)
model.fit(xTrain, yTrain)
yPred = model.predict(xTest)
# Summary of the predictions made by the classifier
print(classification_report(yTest, yPred))
# Accuracy score
print('The accuracy of 4-KNN is: ' + str(accuracy_score(yTest, yPred)))

#Support Vector Machine Test
from sklearn import svm
model = svm.SVC() #select the algorithm
model.fit(xTrain, yTrain) # train the algorithm with the training data and the training output
yPred=model.predict(xTest) #pass the testing data to the trained algorithm
# Accuracy score
print(classification_report(yTest, yPred))
print('The accuracy of the SVM is: ' + str(accuracy_score(yTest, yPred)))

#keep track of results across several trials
logisticAccuracy = []
knn2Accuracy = []
knn4Accuracy = []
svmAccuracy = []
#set number of trials
iterations = 1000
#rerun the previous process without print statements to accumulate more data
for i in range(iterations):
    ptrain, ptest = train_test_split(data, test_size = 0.5) #creates two data sets, where ptrain is .5 of the data points and ptest is (1-.5) of the data points
    xTrain= ptrain[['sepal_length','sepal_width','petal_length','petal_width']]
    yTrain= ptrain.classification
    xTest= ptest[['sepal_length','sepal_width','petal_length','petal_width']]
    yTest= ptest.classification

    model = LogisticRegression()
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    logisticAccuracy.append(accuracy_score(yTest, yPred))

    model = KNeighborsClassifier(n_neighbors=2)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    knn2Accuracy.append(accuracy_score(yTest, yPred))

    model = KNeighborsClassifier(n_neighbors=4)
    model.fit(xTrain, yTrain)
    yPred = model.predict(xTest)
    knn4Accuracy.append(accuracy_score(yTest, yPred))

    model = svm.SVC()
    model.fit(xTrain, yTrain)
    yPred=model.predict(xTest)
    svmAccuracy.append(accuracy_score(yTest, yPred))

#sum up accuracies and take the average
logMean = 0
knn2Mean = 0
knn4Mean = 0
svmMean = 0
for i in range(iterations):
    logMean += logisticAccuracy[i]
    knn2Mean += knn2Accuracy[i]
    knn4Mean += knn4Accuracy[i]
    svmMean += svmAccuracy[i]
logMean = logMean/iterations
knn2Mean = knn2Mean/iterations
knn4mean = knn4Mean/iterations
svmMean = svmMean/iterations
#print final results
print('The average accuracy of Logistic Regression over 1000 trials is: ' + str(logMean))
print('The average accuracy of 2-Nearest Neighbors over 1000 trials is: ' + str(knn2Mean))
print('The average accuracy of 4-Nearest Neighbors over 1000 trials is: ' + str(knn4mean))
print('The average accuracy of Support Vector Machine over 1000 trials is: ' + str(svmMean))