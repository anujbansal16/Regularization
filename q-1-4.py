#!/usr/bin/env python
# coding: utf-8

# # <p style="padding:15px;background:yellow;text-align:center;text-decoration:underline">Assigment 7<p>

# ### - Import Libraries
# 
# Import necessary libraries used in these assignment.

# In[111]:


import numpy as np
import pandas as pd
import math
import sys
import operator
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.model_selection import KFold


# ### - Read File (without header) given a delimeter
# 
# Reads a file with a given delimeter and returns the converted numpy array.
# <br>
# Default when no delimeter is given it reads file as in csv format.

# In[2]:


def readFile(trainFile,seperator=",",sample=False):
    try:
        data=pd.read_csv(trainFile, sep=seperator, header=None)
        if sample:
            data=data.sample(frac=1)
        return data.values
    except:
        print("Error reading training data file")


# ### -Train Test split
# 
# Given a percentage, it will split out the data into train dataset and test dataset.

# In[3]:


def splitTrainTest(data,percent):
    total=len(data)
    trainTotal=int(total*percent*0.01)
    testTotal=total-trainTotal
    return (data[0:trainTotal],data[trainTotal:total])


# ### -Mean square error as error measure
# 
# Following **Error Measure** have been used to perform the various kinds of regressions.
# 
# <!--**Mean Absolute Error:**
# $$ MAE = \frac{1}{n}\sum_{\forall y}|y_{actual}-y_{predicted} | $$
# <hr>-->
# 
# **Mean Square Error:**
# $$ MAE = \frac{1}{n}\sum_{\forall y}(y_{actual}-y_{predicted} ){^2} $$
# 
# <!--<hr>
# **Mean Percentage Error:**
# $$ MAE = \frac{100\%}{n}\sum_{\forall y}\frac{(y_{actual}-y_{predicted} )}{y_{actual}} $$-->
# 

# In[5]:


# function return Mean square error
def MSE(testYs,predictions):
    error=0
    for actual,predicted in zip(testYs,predictions):
        predicted=predicted[0]
        error+=(actual-predicted)**2
#     print("Mean Square Error = ",error/len(testYs))
    return error/len(testYs)




# <hr>
# 
# ## Question-1-Part-4
# ### Analyse how the two different regularisation techniques affect regression weights in terms of their values and what are the differences between the two.
# 
# Below subgraphs shows the effect of different regularization techniques affecting the regression weights for different value of regularization coefficents.
# 
# **We have done the ridge regression using the matrix equations**
# 
# **We have done the lasso regression using the gradient decent**

# In[359]:


def weightLambdaRidge(trainFile,percent,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
    data=pd.read_csv(trainFile).values
#     data=data.sample(frac=1)
    independentVariable=[0]+independentVariable
    features=["GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]

    train,test=splitTrainTest(data,percent)
    
    if testFile:
        print("Will evaluate test File: ",testFile)
        test=pd.read_csv(testFile).values
    else:
        print("Will evaluate Validation data (20%): ")
    
    testY=test[:,targetIndex]
    trainY=train[:,targetIndex]
    
    test[:,0]=1 #changeing first column to constant so it can be used for intercept
    train[:,0]=1
    
    train=train[:,independentVariable]
    test=test[:,independentVariable]
   
    mseErrorsValid=[]
    mseErrorsTrain=[]
#     lambdaVal=[i for i in range(0,10)]
    lambdaVal=[0,0.5,1,5,10,20,50]
    coeffList=[]
    for i in lambdaVal:
        y=np.transpose(np.matrix(trainY))
        X=np.matrix(train)
        XT=np.transpose(X)
        iden=np.identity(XT.shape[0])*i
        iden[0][0]=0
        inverse=np.linalg.inv(XT*X+iden)
        coefficents=inverse*XT*y
        predictedValid=np.array(test*coefficents)
        predictedTrain=np.array(train*coefficents)
        mseErrorsValid.append(MSE(testY,predictedValid))
        mseErrorsTrain.append(MSE(trainY,predictedTrain))
        coeffList.append(coefficents.reshape(1,8).tolist()[0][1:])
    
    fig=plt.figure(figsize=(16, 4))
    i=1
    for coeff,lamVal in zip(coeffList,lambdaVal):  
        ax=fig.add_subplot( 1, len(lambdaVal),i)
        ax.set_title("Lambda= "+str(lamVal))
        ax.grid(True)
        ax.set_xlabel("Features")
        ax.set_ylabel("Coefficents")
        plt.bar(features, coeff)
        i+=1
        plt.xticks(features, rotation=90)
    plt.show()
        



def weightLambdaRidge(trainFile,percent,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
    data=pd.read_csv(trainFile).values
#     data=data.sample(frac=1)
    independentVariable=[0]+independentVariable
    features=["GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]

    train,test=splitTrainTest(data,percent)
    
    if testFile:
        print("Will evaluate test File: ",testFile)
        test=pd.read_csv(testFile).values
    else:
        print("Will evaluate Validation data (20%): ")
    
    testY=test[:,targetIndex]
    trainY=train[:,targetIndex]
    
    test[:,0]=1 #changeing first column to constant so it can be used for intercept
    train[:,0]=1
    
    train=train[:,independentVariable]
    test=test[:,independentVariable]
   
    mseErrorsValid=[]
    mseErrorsTrain=[]
#     lambdaVal=[i for i in range(0,10)]
    lambdaVal=[0,0.5,1,5,10,20,50]
    coeffList=[]
    for i in lambdaVal:
        y=np.transpose(np.matrix(trainY))
        X=np.matrix(train)
        XT=np.transpose(X)
        iden=np.identity(XT.shape[0])*i
        iden[0][0]=0
        inverse=np.linalg.inv(XT*X+iden)
        coefficents=inverse*XT*y
        predictedValid=np.array(test*coefficents)
        predictedTrain=np.array(train*coefficents)
        mseErrorsValid.append(MSE(testY,predictedValid))
        mseErrorsTrain.append(MSE(trainY,predictedTrain))
        coeffList.append(coefficents.reshape(1,8).tolist()[0][1:])
    
    fig=plt.figure(figsize=(16, 4))
    fig.suptitle('Ridge Regression effect of lambda on regression weights', fontsize=12)
    fig.text(0.04, 0.5, 'Coefficents', va='center', rotation='vertical')
    i=1
    for coeff,lamVal in zip(coeffList,lambdaVal):  
        ax=fig.add_subplot( 1, len(lambdaVal),i)
        ax.set_title("Lambda= "+str(lamVal))
        ax.grid(True)
        ax.set_xlabel("Features")
#         ax.set_ylabel("Coefficents")
        plt.bar(features, coeff)
        i+=1
        plt.xticks(features, rotation=90)
    plt.show()
        

def weightLambdaLasso(trainFile,percent,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
    data=pd.read_csv(trainFile).values
#     data=data.sample(frac=1)
    independentVariable=[0]+independentVariable
    features=["GRE Score","TOEFL Score","University Rating","SOP","LOR" ,"CGPA","Research"]
    train,test=splitTrainTest(data,percent)
    
    if testFile:
        print("Will evaluate test File: ",testFile)
        test=pd.read_csv(testFile).values
    else:
        print("Will evaluate Validation data (20%): ")
    
    
    testY=test[:,targetIndex]
    trainY=train[:,targetIndex]
    
    train=(train-train.mean(axis=0))/(train.std(axis=0))
    test=(test-test.mean(axis=0))/(test.std(axis=0))
    
    test[:,0]=1 #changeing first column to constant so it can be used for intercept
    train[:,0]=1
    
    train=train[:,independentVariable]
    
    test=test[:,independentVariable]

        
    mseErrorsValid=[]
    mseErrorsTrain=[]
#     lambdaVal=[i for i in range(5)]
    lambdaVal=[0,0.5,1,5,10,20,50]
#     lambdaVal=np.linspace(1,20,200)
    coeffList=[]
    trainY=trainY.reshape(-1,1)
    testY=testY.reshape(-1,1)
    m=len(trainY)
    for lambdaV in lambdaVal:
        learningR=0.01
        epochs=1000
        theta=np.zeros((train.shape[1],1))
        
        for i in range(epochs):
            predictions=np.dot(train,theta)
            diff=(predictions-trainY)
            temp=theta[0]
#             deri=(train.T.dot(diff))*2/m+lambdaV*np.sign(theta)
            deri=((train.T.dot(diff))+lambdaV*np.sign(theta))
            theta=theta-learningR*(deri)*2/m
            theta[0]=theta[0]+learningR*lambdaV*np.sign(temp)*2/m
        predictedTrain=np.dot(train,theta)
        predictedValid=np.dot(test,theta)
        mseErrorsValid.append(MSE(testY,predictedValid))
        mseErrorsTrain.append(MSE(trainY,predictedTrain))
        coeffList.append(theta.reshape(1,8).tolist()[0][1:])
    fig=plt.figure(figsize=(16, 4))
    fig.suptitle('Lasso Regression effect of lambda on regression weights', fontsize=12)
    fig.text(0.04, 0.5, 'Coefficents', va='center', rotation='vertical')
    i=1
    for coeff,lamVal in zip(coeffList,lambdaVal):  
        ax=fig.add_subplot( 1, len(lambdaVal),i)
        ax.set_title("Lambda= "+str(lamVal))
        ax.grid(True)
        ax.set_xlabel("Features")
#         ax.set_ylabel("Coefficents")
        plt.bar(features, coeff)
        i+=1
        plt.xticks(features, rotation=90)
    plt.show()


testFile=None
if len(sys.argv)>1:
    testFile=sys.argv[1]

print("Ridge Regression effect of lamda on regression weights")
weightLambdaRidge("AdmissionDataset/data.csv",80,[1,2,3,4,5,6,7],8,False,testFile)
print("Lasso Regression effect of lamda on regression weights")
weightLambdaLasso("AdmissionDataset/data.csv",80,[1,2,3,4,5,6,7],8,False,testFile)