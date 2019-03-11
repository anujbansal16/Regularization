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
# ## Question-1-Part-5
# ### k-Fold Cross Validation
# 
# Cross-validation is a resampling procedure used to evaluate machine learning models on a limited data sample.
# 
# It estimate the skill of a machine learning model on unseen data. That is, to use a limited sample in order to estimate how the model is expected to perform in general when used to make predictions on data not used during the training of the model.
# 
# The general procedure is as follows:
# 
# 1. Shuffle the dataset randomly.
# 2. Split the dataset into k groups
# 3. For each unique group:
#     1. Take the group as a hold out or test data set
#     2. Take the remaining groups as a training data set
#     3. Fit a model on the training set and evaluate it on the test set
#     4. Retain the evaluation score and discard the model
# 4. Summarize the skill of the model using the sample of model evaluation scores
# 
# **Three common tactics for choosing a value for k are as follows:**
# 
# 1. **Representative:** The value for k is chosen such that each train/test group of data samples is large enough to be statistically representative of the broader dataset.
# 2. **k=10:** The value for k is fixed to 10, a value that has been found through experimentation to generally result in a model skill estimate with low bias a modest variance.
# 3. **k=n:** The value for k is fixed to n, where n is the size of the dataset to give each test sample an opportunity to be used in the hold out dataset. This approach is called leave-one-out cross-validation.
# 
# #### - How behavior changes with different values of k ?
# 
# - When `k` is small, say `(k=5)` we are removing a much larger chunk of our data $(20\%)$ each time, so our model has a much smaller amount of data to "learn from" hence we get a **large** error.
# - When `k` is large, say `(k=30)` we are removing a relatively smaller chunk of our data $(3.33\%)$ each time. Thus our model has a much better chance of picking up all the relevant "structure" in the training part when `k` is large, hence we get a **small** error. 
# - So When k is small, there is a larger chance that the "left out" part will contain a structure which is absent from the "left in" sub-sample.
# 
# - Also a lower value of k is more biased, and hence undesirable. On the other hand, a higher value of K is less biased, but can suffer from large variability.<br>Thus again we have to consider some bias-variance kind of tradeoff while choosing the value of k , experimentally it is suggested to keep k in between $5-10$.
# 
# * **Lower K = cheaper, less variance, more bias.**
# * **Higher K = more expensive, more variance, and lower bias.**

# In[214]:


def kFold(trainFile,percent,lambdaV=10,mxfolds=30,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
    data=pd.read_csv(trainFile)
    data=data.sample(frac=1).values
    independentVariable=[0]+independentVariable
    
    train,testFinal=splitTrainTest(data,percent)
    if testFile:
        testFinal=pd.read_csv(testFile).values
    
    testFinalY=testFinal[:,targetIndex]
    testFinal[:,0]=1 #changeing first column to constant so it can be used for intercept
    train[:,0]=1
    testFinal=testFinal[:,independentVariable]
    ######################### K-Folds #####################
#     kValues=[i for i in range(2,len(train)+1)]
    kValues=[i for i in range(2,mxfolds)]
    kFoldAvgError=[]
    kFoldUnseenErr=[]
    for k in kValues:
#         print(k)
        mseErrorsValid=[]
        mseErrorsTrain=[]
        
        kf = KFold(n_splits=k)
        kf.get_n_splits(train)
        for train_index, test_index in kf.split(train):
            testY=train[test_index,targetIndex]
            trainY=train[train_index,targetIndex]

            trainX=train[train_index]
            trainX=trainX[:,independentVariable]
            testX=train[test_index]
            testX=testX[:,independentVariable]

            y=np.transpose(np.matrix(trainY))
            X=np.matrix(trainX)
            XT=np.transpose(X)
            iden=np.identity(XT.shape[0])*lambdaV
            iden[0][0]=0
            inverse=np.linalg.inv(XT*X+iden)
            coefficents=inverse*XT*y
            predictedValid=np.array(testX*coefficents)
            predictedUnseen=np.array(testFinal*coefficents)
            mseErrorsValid.append(MSE(testY,predictedValid))

        kFoldAvgError.append(np.average(mseErrorsValid))
        kFoldUnseenErr.append(MSE(testFinalY,predictedUnseen))
    ######################### K-Folds #####################
    print("Optimal value of K: ",kValues[np.argmin(kFoldAvgError)])
    plt.figure(figsize=(8,6))
    plt.xlabel('xlabel', fontsize=15)
    plt.ylabel('ylabel', fontsize=15)
    
    plt.title("Errors (MSE) vs k-folds",fontsize=15)
    plt.ylabel("Avg Error (MSE)")
    plt.xlabel("Folds")
    plt.grid(True)
    plt.plot(kValues,kFoldAvgError,color="orange",linewidth="2.4",label="Validation")
#     plt.plot(kValues,kFoldUnseenErr,color="green",linewidth="2.4",label="Training")
    plt.legend(loc=0,fontsize=15)
    plt.show()


# In[364]:


print("K-fold using ridge's regression")
kFold("AdmissionDataset/data.csv",80,10)


# ### Leave-one-out cross-validation (LOOCV)
# 
# In this approach, we reserve only one data point from the available dataset, and train the model on the rest of the data. This process iterates for each data point. This also has its own advantages and disadvantages. Let’s look at them:
# 
# * We make use of all data points, hence the bias will be low
# * We repeat the cross validation process n times (where n is number of data points) which results in a higher execution time
# * This approach leads to higher variation in testing model effectiveness because we test against one data point. So, our estimation gets highly influenced by the data point. If the data point turns out to be an outlier, it can lead to a higher variation. <br>
# * This can be easily observed from the graph drawn below showing the prediction error of each of the example(when used to validate) as it leads to a high variation compared to that of average error in k-fold cross validation.
# 
# **Gives high variance, may lead to overfitting over the training data**

# In[346]:


def leaveOneOut(trainFile,percent,lambdaV=10,independentVariable=[1,2,3,4,5,6,7],targetIndex=8,forGraph=False,testFile=None):
    data=pd.read_csv(trainFile)
    data=data.sample(frac=1).values
    independentVariable=[0]+independentVariable
    
    train,testFinal=splitTrainTest(data,percent)
    
    if testFile:
        testFinal=pd.read_csv(testFile).values
    
    train[:,0]=1
    plt.figure(figsize=(8,6))
    ######################### K-Folds #####################

    kValues=[len(train)]
    kFoldAvgError=[]
    for k in kValues:
#         print(k)
        mseErrorsValid=[]
        mseErrorsTrain=[]
        
        kf = KFold(n_splits=k)
        kf.get_n_splits(train)
        
        for train_index, test_index in kf.split(train):
            testY=train[test_index,targetIndex]
            trainY=train[train_index,targetIndex]
            trainX=train[train_index]
            trainX=trainX[:,independentVariable]
            testX=train[test_index]
            testX=testX[:,independentVariable]

            y=np.transpose(np.matrix(trainY))
            X=np.matrix(trainX)
            XT=np.transpose(X)
            iden=np.identity(XT.shape[0])*lambdaV
            iden[0][0]=0
            inverse=np.linalg.inv(XT*X+iden)
            coefficents=inverse*XT*y
            predictedValid=np.array(testX*coefficents)
            mse=MSE(testY,predictedValid)
            mseErrorsValid.append(mse)
            plt.plot([i for i in range(100)],[mse]*100,color="cyan",linewidth="2.4")
        kFoldAvgError.append(np.average(mseErrorsValid))
    ######################### K-Folds #####################
    print("Average Error(MSE): ",kFoldAvgError)
    
#     plt.xlabel('xlabel', fontsize=15)
    plt.ylabel('ylabel', fontsize=15)
    plt.title("Leave one out cross validation",fontsize=15)
    plt.ylabel("Error (MSE)")
#     plt.xlabel("Folds")
    plt.grid(True)
    plt.plot([i for i in range(100)],kFoldAvgError*100,color="black",linewidth="2.4",label="Average Error")
#     plt.plot(lambdaVal,mseErrorsTrain,color="green",linewidth="2.4",label="Training")
    plt.legend(loc=0,fontsize=15)
    plt.show()


testFile=None
if len(sys.argv)>1:
    testFile=sys.argv[1]


leaveOneOut("AdmissionDataset/data.csv",80,10,[1,2,3,4,5,6,7],8,False,testFile)


# #### Error(MSE) for each of the training example when used for testing in LOOCV represented by horizontal lines (cyan color) and average of all the errors shown by black horizotal line.
