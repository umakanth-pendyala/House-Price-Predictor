import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import linear_model
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


df = pd.read_csv('house_data.csv')
df = df.replace({'No_Data':0}) 
df["bedrooms"] = pd.to_numeric(df["bedrooms"])
df["sqft_living"] = pd.to_numeric(df["sqft_living"])
df["condition"] = pd.to_numeric(df["condition"])
df["grade"] = pd.to_numeric(df["grade"]) 

df1 = df[["bedrooms","sqft_living","condition","grade"]]

# line added 
# df = df[["bedrooms","sqft_living","condition","grade", "price"]]


meanbr = (round(df1.mean()[0])) 
meansftl = (round(df1.mean()[1]))
meancndn = (round(df1.mean()[2]))
meangr = (round(df1.mean()[3]))


df["bedrooms"] = df["bedrooms"].replace({0:meanbr})
df["bedrooms"] = pd.to_numeric(df["bedrooms"])

df["sqft_living"] = df["sqft_living"].replace({0:meansftl})
df["sqft_living"] = pd.to_numeric(df["sqft_living"])

df["condition"] = df["condition"].replace({0:meancndn})
df["condition"] = pd.to_numeric(df["condition"])

df["grade"] = df["grade"].replace({0:meangr})
df["grade"] = pd.to_numeric(df["grade"])






data = df[["bedrooms", "sqft_living","condition","grade", "price"]].to_numpy()

inputs = data[:,:-1]
outputs = data[:, -1]

training_inputs = inputs[:10000]
training_outputs = outputs[:10000]
testing_inputs = inputs[10000:]
testing_outputs = outputs[10000:]


def plotGraph(color = 'red', name=''):
    plt.scatter(testing_outputs, y_pred,color=color)
    plt.title('Actual Price Vs Predicted Price ' + name, fontsize=14)
    plt.xlabel('Actual Price', fontsize=14)
    plt.ylabel('Predicted Price', fontsize=14)
    plt.grid(True)
    plt.show()


def performGaussianNaiveBayes(bedrooms=0, sqftLiving=0, condition=0, grade=0):
    testset = [[bedrooms, sqftLiving, condition, grade]]
    testset = pd.DataFrame(testset)
    gaussianClassifer = GaussianNB()
    gaussianClassifer.fit(training_inputs, training_outputs)
    return gaussianClassifer.predict(testing_inputs)


def performBaggingRegression():
    bClassifier = BaggingClassifier()
    bClassifier.fit(training_inputs, training_outputs)
    return bClassifier.predict(testing_inputs)


def performDecisionTreeClassifier():
    gClassifier = DecisionTreeClassifier()
    gClassifier.fit(training_inputs, training_outputs)
    return gClassifier.predict(testing_inputs)

def performLinearRegression():
    regr = linear_model.LinearRegression()
    regr.fit(training_inputs, training_outputs)
    return regr.predict(testing_inputs)

def perfromLogesticRegression():
    lClassifer = LogisticRegression()
    lClassifer.fit(training_inputs, training_outputs)
    return lClassifer.predict(testing_inputs)


y_pred = performGaussianNaiveBayes()

plotGraph(name='GNB')

y_pred = performBaggingRegression()

plotGraph('blue', 'BGR')

y_pred = performDecisionTreeClassifier()

plotGraph('green', 'DTC')

y_pred = performLinearRegression()

plotGraph('orange', 'SLR')

y_pred = perfromLogesticRegression()

plotGraph('indigo', 'Logestic')

# for i in range(100):
#     print(training_inputs[i], end = " ")
#     print(training_outputs[i])


# x = df.iloc[:, :-1].values
# y = df.iloc[:, -1].values
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 1000, random_state = 5)

# sc = StandardScaler()  
# x_train = sc.fit_transform(x_train)  
# x_test = sc.transform(x_test) 