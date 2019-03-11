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

# In[371]:





# <hr>
# 
# ## Question-1-Part-2 
# ### Ridge regression
# 
# As we know to calculate the value of dependent variable $Y$ we can use the below general equation:
# 
# $$Y=\theta_{0}+\theta_{1}X_1+\theta_{2}X_2+....+\theta_{n}X_n+\epsilon$$
# 
# In ordinary least squares, the regression coefficients are estimated using the formula matrix calculus
# $$\theta=((X{^T}X)^{-1}X{^T}Y)$$
# 
# But complicated hypothesis may lead to the overfitting which affect the performance of our model on validation or unseen data.
# 
# Thus the variation of linear regression i.e. **Ridge regression** is used which penalizes the hypothesis complexity by using a tradeoff parameter $\lambda $ to prevent overfitting.  This is called regularization and the parameter $\lambda $ is called regularization coefficent.<br>
# Thus we will find the optimal values of $\theta$ by minimising the new Cost function:
# 
# $$ \theta^{Ridge} = \arg\min_{\theta}  \left\{ || X\theta - y ||_2^2 + \lambda || \theta ||_2^2 \right\} $$
# 
# In matrix form we will get the $\theta$ for ridge regression as:
# 
# <hr>
# 
# $$\theta=((X{^T}X + \lambda I)^{-1}X{^T}Y)$$
# 
# <hr>
# 
# While calculating the regularization we dont consider the weight of coefficent corresponding to bias therefore we make the first digonal element of correlation matrix , $I$, zero
# 

# In[370]:


def linearRegressionRidge(trainFile,percent,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
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
    
    test[:,0]=1 #changeing first column to constant so it can be used for intercept
    train[:,0]=1
    
    train=train[:,independentVariable]
    test=test[:,independentVariable]
   
    mseErrorsValid=[]
    mseErrorsTrain=[]
    lambdaVal=[i for i in range(0,200)]
#     lambdaVal=np.linspace(0,1,100)
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
    
    plt.figure(figsize=(8,6))
    plt.xlabel('xlabel', fontsize=15)
    plt.ylabel('ylabel', fontsize=15)
    plt.xlabel("regularisation coefficient λ")
    plt.ylabel("Error (MSE)")
    plt.title("Ridge regression",fontsize=15)
    plt.grid(True)
    plt.plot(lambdaVal,mseErrorsValid,color="orange",linewidth="2.4",label="Validation")
    plt.plot(lambdaVal,mseErrorsTrain,color="green",linewidth="2.4",label="Training")
    print("Optimal value of lambda (for Validation): ",lambdaVal[np.argmin(mseErrorsValid)],"with error ",mseErrorsValid[np.argmin(mseErrorsValid)])
    plt.legend(loc=4,fontsize=15)
    plt.show()


# In[369]:

testFile=None
if len(sys.argv)>1:
    testFile=sys.argv[1]
linearRegressionRidge("AdmissionDataset/data.csv",80,[1,2,3,4,5,6,7],8,False,testFile)