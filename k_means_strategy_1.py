
# coding: utf-8

# In[1]:


#####################################################################################
# Assignment to implement K-Means Algorithm                                         # 
# Course - CSE 575, Statistical Machine Learning                                    #
# Problem Sum - Implement K-means algorithm on the given data set using strategy 1  #
# Strategy 1 - Randomly select centroids from given data set                        #  
# Author - Abhik Dey(akd)                                                           #
#####################################################################################

import scipy.io as spio
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import math


# In[2]:


#Initializing Color Map
color_map = ['red','green', 'blue', 'yellow', 'cyan', 'grey', 'purple', 'maroon', 'yellowgreen', 'skyblue', 'wheat', 'pink', 'orange']


# In[3]:


def extractData():
    """Extract data from Matlab"""
    #####################################################################################
    # Function - extractData()                                                          # 
    # Parameters - NULL                                                                 #
    # Functionality - Extract data from the matlab file                                 #
    # Author - Abhik Dey                                                                #
    #####################################################################################
    mat = spio.loadmat("AllSamples.mat", squeeze_me = True)
    data = mat['AllSamples'] #Taking values in variable from mat
    
    return data


# In[4]:


def initialize_centroids(k,data_set):
    """Initializing centroids at random"""
    #####################################################################################
    # Function - initialize_centroids()                                                 # 
    # Parameters - k - Number of clusters, data_set - data from given matlab file       #
    # Functionality - Initialize centroids at random                                    #
    # Author - Abhik Dey                                                                #
    #####################################################################################
    
    random_index = np.random.choice(a.shape[0], k, replace = False)
    centroids = data_set[random_index]
    #print (centroids)
    
    return centroids


# In[5]:


def calEuclideanDist(x_cord,y_cord, x_cent, y_cent):
    """Calculate Euclidean Distance"""    
    #####################################################################################
    # Function - calEuclideanDist()                                                     # 
    # Parameters - x_cord - x-co-ordinate, y_cord - x-co-ordinate                       #
    #              x_cent - centroid x-co-ordinate, y_cent - centroid y-co-ordinate     #
    # Functionality - Calculate Euclidean Distance                                      #
    # Author - Abhik Dey                                                                #
    #####################################################################################
    x_val = (x_cent - x_cord)**2
    y_val = (y_cent - y_cord)**2
    eucDist = math.sqrt(x_val + y_val)
    
    return eucDist


# In[6]:


def calObjectiveFunction(data_set,centroids):
    #####################################################################################
    # Function - calEuclideanDist()                                                     # 
    # Parameters - data_set - data from matlab, centroids - k Clusters                  #
    # Functionality - Calculate Objective Function                                      #
    # Author - Abhik Dey                                                                #
    #####################################################################################
    sum = 0
    obj_val = []
    
    for ds in data_set:
        obj_val.append(((np.linalg.norm((ds-centroids), axis = 0)**2)))
    return np.sum(obj_val)        


# In[7]:


a = extractData() # Store data in a from the matlab file
print (a)

y_cordinate = a.take(1,axis=1) #extract y co-ordinate
x_cordinate = a.take(0,axis=1) #extract x_co-ordinate

k_val = [2,3,4,5,6,7,8,9,10] # initializing the k values from k = 2-10 @akd


# In[8]:


obj_plot = [] #Storing the values for objective function to plot against k @akd
for k in k_val: 
    
    centroids = initialize_centroids(k,a)
    #print (centroids)
    
    x_centroid = centroids.take(0,axis=1).tolist()
    y_centroid = centroids.take(1, axis=1).tolist()
    
    #plot scatter graph for given data set with initial k
    print ("Initial Graph for k = ",k)
    figure = plt.figure(figsize = (5,5))
    plt.scatter(x_cordinate, y_cordinate)
    plt.scatter(x_centroid, y_centroid, marker = '+', s = 100, color='black')
    plt.xlim(-2,10)
    plt.ylim(-2,10)
    plt.show()
    
    print ("Starting k means algo")

    condition = True
    C = np.zeros(len(a))

    while (condition):
                  
        D = []    
        for xy in a: #iterating over all data points

            ED = []
            for cent in centroids: #calculating all the ED among centroids and a particular point 
                ED.append(calEuclideanDist(xy[0],xy[1],cent[0],cent[1]))
            D.append(ED)

        D = np.asarray(D)

        x_centroid = centroids.take(0,axis=1).tolist()
        y_centroid = centroids.take(1,axis=1).tolist()

        #plot scatter graph
        figure1 = plt.figure(figsize = (5,5))
        C = np.argmin(D,axis=1)

        #Update old Centroid
        centroid_old = np.copy(centroids)

        for i in range(k):
            points = np.asarray([a[j] for j in range(len(a)) if C[j]==i])
            #print ("Cluster Number - ",i)
            #print (points)
            x_c = points.take(0,axis=1).tolist()
            y_c = points.take(1,axis=1).tolist()
            #print(color_map[i])
            plt.scatter(x_c, y_c, c=color_map[i])
            centroids[i] = np.mean(points,axis=0) # initialize new centroid@akd                                                                                                                                   @akd
        plt.scatter(x_centroid, y_centroid, marker = '+', s=100 ,color = "black")
        plt.xlim(-2,10)
        plt.ylim(-2,10)
        plt.show()

       
        condition1 = (np.array_equal(centroid_old, centroids))
        #print (condition1)

        if condition1:
            condition = False
            #print ("Will exit loop")
        else:
            condition = True
            #print ("Loop continues")
    
    #Calculating the objective function and store value to obj_plot to plot against k (Elbow Plot) - @akd                                                                                                                            @akd
    sum_of_clusters = 0
    for i in range(k):
        points = np.asarray([a[j] for j in range(len(a)) if C[j]==i])        
        val = calObjectiveFunction(points,centroids[i])
        sum_of_clusters += val
    print ("Sum of Clusters as whole",sum_of_clusters)
        
    obj_plot.append(sum_of_clusters)
    print (obj_plot)


# In[9]:


#Plot the Elbow Plot (X-> k, Y-> Objective Function) - @akd
plt.plot(k_val, obj_plot, 'bx-') 
plt.xlabel('Values of K') 
plt.ylabel('Objective Function') 
plt.title('Elbow Plot') 
plt.show()

