import numpy as np
import pandas as pd
from sklearn import *
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn import linear_model
from sklearn.ensemble import BaggingRegressor
from sklearn.linear_model import LogisticRegression
import statsmodels.api as sm


df = pd.read_csv('house_data.csv')
df = df.replace({'No_Data':0}) 
df["bedrooms"] = pd.to_numeric(df["bedrooms"])
df["sqft_living"] = pd.to_numeric(df["sqft_living"])
df["condition"] = pd.to_numeric(df["condition"])
df["grade"] = pd.to_numeric(df["grade"]) 

df1 = df[["bedrooms","sqft_living","condition","grade"]]
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

data = df[["bedrooms","sqft_living","condition","grade","price"]].to_numpy()
inputs = data[:,:-1]
outputs = data[:, -1]
training_inputs = inputs[:2000]
training_outputs = outputs[:2000]
testing_inputs = inputs[2000:]
testing_outputs = outputs[2000:]

def getDTC(bedrooms, sqftLiving, condition, grade):
    testSet = [[bedrooms, sqftLiving, condition, grade]]
    test = pd.DataFrame(testSet)
    classifier = DecisionTreeClassifier()
    classifier.fit(training_inputs, training_outputs)
    predictions = classifier.predict(testing_inputs)
    return classifier.predict(test)

def getGNB(bedrooms, sqftLiving, condition, grade):
    testSet = [[bedrooms, sqftLiving, condition, grade]]
    test = pd.DataFrame(testSet)
    classifier = GaussianNB()
    classifier.fit(training_inputs, training_outputs)
    predictions = classifier.predict(testing_inputs)
    return classifier.predict(test)

def getRGRp(bedrooms, sqftLiving, condition, grade):
    testSet = [[bedrooms, sqftLiving, condition, grade]]
    test = pd.DataFrame(testSet)
    regr = linear_model.LinearRegression()
    regr.fit(training_inputs, training_outputs)
    return regr.predict(test)

def getBaggingRegression(bedrooms, sqftLiving, condition, grade):
    testSet = [[bedrooms, sqftLiving, condition, grade]]
    test = pd.DataFrame(testSet)
    regressionModel = BaggingRegressor()
    regressionModel.fit(training_inputs, training_outputs)
    return regressionModel.predict(test)

def getLogisticRegression(bedrooms, sqftLiving, condition, grade):
    testSet = [[bedrooms, sqftLiving, condition, grade]]
    test = pd.DataFrame(testSet)
    lClassifer = LogisticRegression()
    lClassifer.fit(training_inputs, training_outputs)
    return lClassifer.predict(test)



# 4 1800 4 6
