import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
import random
import math

# ========================================================================
# dataset

n_tot = 400
n = int(n_tot/2)
# two blobs, not completely separated
#X, y = make_blobs(n_tot, centers=2, cluster_std=3.0, random_state=2)
X, y = make_moons(n_tot, noise=0.15, random_state=0)
#plt.figure()

colors = ["g", "b"]
for ii in range(2):
    class_indices = np.where(y==ii)[0]
    plt.scatter(X[class_indices, 0], X[class_indices, 1], c=colors[ii])
plt.title("full dataset")
#plt.show()

# divide data into training and testing
# NOTE! Test data is not needed in solving the exercise

np.random.seed(42)
order = np.random.permutation(n_tot)
train = order[:n] # take half of data to train
# test = order[100:]

Xtr = X[train, :] #coordinate of training data
ytr = y[train] #train has 200 data so ytr will generate 200 random 0,1 values

# Xtst = X[test, :]
# ytst = y[test]

# ========================================================================
# classifier

# The perceptron algorithm will be encountered later in the course
# How exactly it works is not relevant yet, it's enough to just know it's a binary classifier
from sklearn.linear_model import Perceptron as binary_classifier

# # It can be used like this:
bc = binary_classifier()
# bc.fit(Xtr, ytr)  # train the classifier on training data
# preds = bc.predict(Xtst)  # predict with test data

# ========================================================================
# setup for analysing the Rademacher complexity

# consider these sample sizes
print_at_n = [20, 50, 100]
# when analysing Rademacher complexity, take always n first samples from training set, n as in this array

#Classifier

def cal_gen_bound(m):
    Xtraining = Xtr[:m,:] #the first m samples of the training set
    Ytraining = ytr[:m]
    Yrandom = np.zeros(m) #Defining an array for random labels
    
   
    emp_risk = cal_emp_risk(m,  Xtraining, Ytraining)
    rademacher_risk = cal_rademacher_risk(m, Xtraining, Yrandom)
    
    Term = 3 * math.sqrt((math.log(2/0.05))/(2*m))
    R = emp_risk + rademacher_risk + Term #Generalization bound
    return R


def cal_emp_risk (m, Xtraining, Ytraining):
    count = 0
    bc.fit(Xtraining, Ytraining) #Now with training samples, not random labels
    Ypred = bc.predict(Xtraining) #Seeing how good is the model
    
    for j in range(0, m):
        if Ypred[j] != Ytraining[j]: #Seeing if these match or not and counting them
            count += 1
        emp_risk = count/m
        # print("Count", j, count, m, emp_risk)
    return emp_risk

def cal_rademacher_risk(m, Xtraining, Yrandom):
    count = 0
    Epsilon = np.zeros(200)#Defining an array for empirical risk
    Total_epsilon = 0
    for rango in range(0, 200): #To calculate the average risks, 200 samples of random sets
        for length in range(0,m):
            Yrandom[length] = random.randint(0, 1) #Preparing the random labels
        bc.fit(Xtraining, Yrandom) #Creating the model
        Ypred = bc.predict(Xtraining) #Seeing how well
        
        for j in range(0, m): #Comparing both labels to count
            if Ypred[j] != Yrandom[j]:
                count += 1 #Counting
        Epsilon[rango] = count/m #Calculating empirical risk for each M sample
        count = 0
        
    for each in range(0, 200):
        Total_epsilon += Epsilon[each]
    
    return (1/2) - Total_epsilon/200 #Second term of Rademacher

for i in print_at_n:
    print("Rademacher " + str(i) + " samples: " + str(cal_gen_bound(i)))

