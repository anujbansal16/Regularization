#!/usr/bin/env python
# coding: utf-8

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
# ## Question-1-Part-1
# ### Lasso regression
# 
# As we know to calculate the value of dependent variable $Y$ we can use the below general equation:
# 
# $$Y=\theta_{0}+\theta_{1}X_1+\theta_{2}X_2+....+\theta_{n}X_n+\epsilon$$
# 
# But complicated hypothesis may lead to the overfitting which affect the performance of our model on validation or unseen data.
# 
# Thus the variation of linear regression i.e. **Lasso regression** is used which penalizes the hypothesis complexity by using a tradeoff parameter $\lambda $ to prevent overfitting.  This is called regularization and the parameter $\lambda $ is called regularization coefficent.<br>
# Thus we will find the optimal values of $\theta$ by minimising the new Cost function:
# 
# $$ \theta^{Lasso} = \arg\min_{\theta}  \left\{ || X\theta - y  ||_2^2 + \lambda || \theta ||_1 \right\} $$
# 
# Thus our new cost function which we have to minimize becomes:
# 
# $$J(\theta) =  || X\theta - y ||_2^2 + \lambda || \theta ||_1  $$
# 
# or,
# 
# $$J(\theta) =  ( X\theta - y)^{T}(X\theta - y ) + \lambda\theta   $$
# 
# Then we minimize above cost function by **gradient decent algorithm** to find optimal $\theta$.<br>
# 
# $$\frac{\partial{J}}{\partial{\theta}} = 2X^{T}y + 2X^{T}X{\theta} + \lambda sign(\theta)$$
# 
# 
# While calculating the regularization we dont consider the weight of coefficent corresponding to bias.
# 

# In[357]:


def linearRegressionLasso(trainFile,percent,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
    data=pd.read_csv(trainFile)
    data=data.sample(frac=1).values
    independentVariable=[0]+independentVariable
    
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
#     lambdaVal=[i for i in range(50)]
    lambdaVal=np.linspace(1,50,500)
    
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
#             theta=theta-(1/m)*learningR*(train.T.dot((predictions-trainY))+lambdaV*np.sign(theta))
        predictedTrain=np.dot(train,theta)
        predictedValid=np.dot(test,theta)
        mseErrorsValid.append(MSE(testY,predictedValid))
        mseErrorsTrain.append(MSE(trainY,predictedTrain))
    print("Optimal value of lambda: ",lambdaVal[np.argmin(mseErrorsValid)])
    plt.figure(figsize=(8,6))
    plt.xlabel('xlabel', fontsize=15)
    plt.ylabel('ylabel', fontsize=15)
    plt.xlabel("regularisation coefficient λ")
    plt.ylabel("Error (MSE)")
    plt.grid(True)
    plt.plot(lambdaVal,mseErrorsValid,color="orange",linewidth="2.4",label="Validation")
    plt.plot(lambdaVal,mseErrorsTrain,color="green",linewidth="2.4",label="Training")
    plt.title("Lasso regression",fontsize=15)
    plt.legend(loc=0,fontsize=15)
    plt.show()



testFile=None
if len(sys.argv)>1:
    testFile=sys.argv[1]
linearRegressionLasso("AdmissionDataset/data.csv",80,[1,2,3,4,5,6,7],8,False,testFile)