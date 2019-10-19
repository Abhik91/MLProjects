
# coding: utf-8

# In[202]:


######################################################################################################################
# Project - Density Estimation and Classification
# Subject - CSE575
# Author -  Abhik Dey (ASU Id - 1216709406)
# Goal is:
#    1. Extracting features and parameter estimation for the normal distribution of each digit using training data.
#    2. Using estimated distributions to do Naive Bayes Classification on testing data.
#    3. Using training data to train the model for Logistic Regression using gradient ascent.
#    4. Calculate the classification accuracy for both “7” and “8”  
######################################################################################################################

get_ipython().magic('matplotlib inline')
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from skimage import io, img_as_float
from skimage.io import imread, imshow
import pylab as pl
import scipy.stats as stats
import math
import itertools


# In[203]:


dataset = scipy.io.loadmat('/home/abhik/Downloads/mnist_data.mat')


# In[204]:


print (dataset)


# In[205]:


training_data_set = dataset['trX']
print (training_data_set)
print ('length of training data set:',len(training_data_set))


# In[206]:


testing_data_set = dataset["tsX"]
print (testing_data_set)
print ('length of training data set:',len(testing_data_set) )


# In[207]:


imgplot = plt.imshow(np.reshape(dataset["trX"][6265],[28,28]),cmap=plt.cm.gray)


# In[208]:


trx_7 = training_data_set[0:6265]
trx_8 = training_data_set[6265:]


# In[209]:


print (len(trx_7))
print (trx_7.shape)


# In[210]:


print (len(trx_8))


# In[211]:


mean_7 = np.mean(trx_7, axis = 1)
print(mean_7)
print(len(mean_7))


# In[212]:


std_7 = np.std(trx_7, axis = 1)
print(std_7)


# In[213]:


mean_of_feature_mean_7 = mean_7.mean()
print (mean_of_feature_mean_7)


# In[214]:


std_of_feature_mean_7 = np.std(mean_7)
print (std_of_feature_mean_7)


# In[215]:


mean_of_feature_std_7 = np.mean(std_7)
print (mean_of_feature_std_7)


# In[216]:


std_of_feature_std_7 = np.std(std_7)
print (std_of_feature_std_7)


# In[217]:


graph_mu_7 = stats.norm.pdf(np.sort(mean_7),mean_of_feature_mean_7,std_of_feature_mean_7)
pl.plot(np.sort(mean_7), graph_mu_7)


# In[218]:


graph_sigma_7 = stats.norm.pdf(np.sort(std_7),mean_of_feature_std_7,std_of_feature_std_7)
pl.plot(np.sort(std_7), graph_sigma_7)


# In[219]:


mean_8 = np.mean(trx_8,axis = 1)
print (mean_8)


# In[220]:


std_8 = np.std(trx_8, axis = 1)
print(std_8)


# In[221]:


mean_of_feature_mean_8 = np.mean(mean_8)
print (mean_of_feature_mean_8)


# In[222]:


std_of_feature_std_8 = np.std(mean_8)
print(std_of_feature_std_8)


# In[223]:


mean_of_feature_std_8 = np.mean(std_8)
print (mean_of_feature_std_8)


# In[224]:


std_of_feature_Std_8 = np.std(std_8)
print (std_of_feature_std_8)


# In[225]:


graph_mu_8 = stats.norm.pdf(np.sort(mean_8),mean_of_feature_mean_8,std_of_feature_std_8)
pl.plot(np.sort(mean_8), graph_mu_8)


# In[226]:


graph_sigma_8 = stats.norm.pdf(np.sort(std_8),mean_of_feature_std_8,std_of_feature_std_8)
pl.plot(np.sort(std_8), graph_sigma_8)


# In[227]:


# Derving Mean and Std of Testing Data set
tsX_mean = np.mean(testing_data_set, axis = 1)
tsX_std = np.std(testing_data_set, axis = 1)
print (tsX_mean)
print (tsX_std)


# In[228]:


# Function to derive the likelihood
def p_x_given_y(X, mean, var):
    p = (1/(np.sqrt(2*np.pi*var)))*np.exp(-(X-mean)**2/(2*var))
    return p


# In[229]:


def calNaiveBayes():
    #Calculate posterior for 7
    p_prior_7 = len(trx_7)/len(training_data_set)  
    p_likelihood_7 = p_x_given_y(tsX_mean,mean_of_feature_mean_7,mean_7.var())*                     p_x_given_y(tsX_std, mean_of_feature_std_7,std_7.var())
    p_posterior_7 = p_prior_7*p_likelihood_7
    
    #Calculate posterior for 7
    p_prior_8 = len(trx_8)/len(training_data_set)
    p_likelihood_8 = p_x_given_y(tsX_mean,mean_of_feature_mean_8,mean_8.var())*                     p_x_given_y(tsX_std, mean_of_feature_std_8,std_8.var())
    p_posterior_8 = p_prior_8*p_likelihood_8
    
    #Compare between class 7 & 8
    compare = np.greater(p_posterior_8,p_posterior_7)
    compare_numeric=compare.astype(np.int)
    
    #Validate with tsY
    validate = np.equal(compare_numeric,dataset["tsY"])
    
    #Check for non-zero values in validate
    count = np.count_nonzero(validate)
    
    #compute accuracy
    accuracy = (count/len(testing_data_set))*100
    print ("Accuracy is ",accuracy,"%")   


# In[230]:


calNaiveBayes()


# In[231]:


#Logistic Regression
trX_data = pd.DataFrame({'Mean':np.mean(dataset["trX"],axis=1), 'STD':np.std(dataset["trX"],axis=1)})


# In[232]:


#convert data to array
trX_data_array = trX_data.to_numpy()


# In[233]:


#Test data for Logistic Regression
tsY_data = dataset["trY"]
print(len(tsY_data))
squeezedMatrix=np.squeeze(tsY_data)


# In[240]:


class LogisticRegression:
    def __init__(self, learningRate=0.01, iterations=200000, fit_intercept=True, verbose=False):
        self.learningRate = learningRate
        self.iterations = iterations
        self.fit_intercept = fit_intercept
        self.verbose = verbose
    
    def add_intercept(self, trX):
        intercept = np.ones((trX.shape[0], 1))
        return np.concatenate((intercept, trX), axis=1)
    
    def sigmoid_func(self, z):
        return 1 / (1 + np.exp(-z))
    
    def loss(self, h, y):
        return (-y * np.log(h) - (1 - y) * np.log(1 - h)).mean()
    
    def fit(self, trX, trY):
        if self.fit_intercept:
            trX = self.add_intercept(trX)
        
        # weights initialization
        self.theta = np.zeros(trX.shape[1])
        
        for i in range(self.iterations):
            z = np.dot(trX, self.theta)
            h = self.sigmoid_func(z)
            gradient = np.dot(trX.T, (h - trY)) / trY.size
            self.theta -= self.learningRate * gradient
            
            if(self.verbose == True and i % 10000 == 0):
                z = np.dot(trX, self.theta)
                h = self.sigmoid_func(z)
                print(f'loss: {self.__loss(h, trY)} \t')
    
    def predict_prob(self, trX):
        if self.fit_intercept:
            trX = self.add_intercept(trX)    
        return self.sigmoid_func(np.dot(trX, self.theta))
    
    def predict(self, X):
        return self.predict_prob(X)


# In[251]:


#Taking learning rate as 0.01 and number of iterations as 500000
logisticRegression = LogisticRegression(learningRate=0.01, iterations=500000)


# In[252]:


logisticRegression.fit(trX_data_array, squeezedMatrix)


# In[253]:


test_dataset = pd.DataFrame({'Mean':np.mean(dataset["tsX"],axis=1), 'SD':np.std(dataset["tsX"],axis=1)})
test_np=test_dataset.to_numpy()


# In[254]:


preds=logisticRegression.predict(test_np).round()


# In[255]:


compareLr = np.equal(dataset["tsY"],preds)
countLr = np.count_nonzero(compareLr)

#compute accuracy
accuracy = (countLr/len(testing_data_set))*100
print ("Accuracy is ",accuracy,"%")  

