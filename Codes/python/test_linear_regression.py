# -*- coding: utf-8 -*-


import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as nalg
import warnings
from sklearn.datasets import load_boston
with warnings.catch_warnings():
    warnings.filterwarnings("ignore")

from sklearn import preprocessing

def mean_squared_error(y_true, y_predicted):
    cost = np.sum((y_true-y_predicted)**2) / len(y_true)
    return cost

def TGCR(x, y, iterations = 1000, learning_rate = 0.01,
                     stopping_threshold = 1e-6):
     
    # Initializing weight, bias, learning rate and iterations
    w = np.random.rand(14)
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))
    d = 14
    lb = 3
    epsf = 1
    P = np.zeros((d, lb))
    AP = np.zeros((d,lb))
    def FF(x, y, n, w):
         y_predicted = x @ w
         weight_derivative = -(2/n) * (x.transpose() @ (y-y_predicted))
         return learning_rate * weight_derivative
    r = FF(x, y, n, w)
    rho = nalg.norm(r)
    ep = epsf * nalg.norm(w)/rho
    print((ep*r).shape, w.shape)
    Ar = (FF(x, y, n, w-ep*r)-r)/ep
    t = nalg.norm(Ar)
    t = 1.0/t
    P[:,0] = t*r
    AP[:,0]=  t *Ar
    costs = []
    weights = []
    previous_cost = None
    i2 = 1
    i = 1
    # Estimation of optimal parameters
    for it in range(iterations):
        
        y_predicted = x @w
        current_cost = mean_squared_error(y, y_predicted)
 
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost

        print(f"Iteration {it+1}: Cost {current_cost}")
        alph = np.dot(np.transpose(AP),r)
        
        w = w + P@(alph)
        r = FF(x, y, n, w)
        rho = nalg.norm(r)
        Ar = (FF(x, y, n, w-ep*r)-r)/ep
        ep = epsf * nalg.norm(w)/rho
        p = r
        if i <= lb:
            k = 0
        else:
            k = i2
        while True:
            if k ==lb:
                k = 0
            k +=1
      
            tau = np.inner(Ar, AP[:,k-1])
            
            p = p - tau*(P[:,k-1])
            Ar = Ar -  tau*(AP[:,k-1])
     
            if k == i2:
                break
        t = nalg.norm(Ar)
        # if (it+1)%lb ==0:
        #     i2 =-1
        #     i = -1
        #     P = np.zeros((d, lb))
            
        #     AP = np.zeros((d, lb))
        #     r = FF(x, y, n, w)
        #     rho = nalg.norm(r)
        #     Ar = (FF(x, y, n, w-ep*r)-r)/ep
        #     ep = epsf * nalg.norm(w)/rho
        #     t = nalg.norm(Ar)
        #     p = r
        if (i2) == lb:
            i2 = 0
        i2 = i2+1
        i = i+1
        t = 1.0/t
        AP[:,i2-1] = t*Ar
        P[:,i2-1] = t*p

     
     
    # Visualizing the weights and cost at for all iterations
    plt.figure(figsize = (8,6))
    plt.plot(weights, costs)
    plt.scatter(weights, costs, marker='o', color='red')
    plt.title("Cost vs Weights")
    plt.ylabel("Cost")
    plt.xlabel("Weight")
    plt.show()
     
    return w

def gradient_descent(x, y, iterations = 1000, learning_rate = 0.001,
                     stopping_threshold = 1e-6):
     
    # Initializing weight, bias, learning rate and iterations
    w = np.random.rand(14)
    iterations = iterations
    learning_rate = learning_rate
    n = float(len(x))
    previous_cost = None
    for i in range(iterations):
         
        # Making predictions
        y_predicted = x @w
        current_cost = mean_squared_error(y, y_predicted)
         
        # Calculationg the current cost
        current_cost = mean_squared_error(y, y_predicted)
 
        # If the change in cost is less than or equal to
        # stopping_threshold we stop the gradient descent
        if previous_cost and abs(previous_cost-current_cost)<=stopping_threshold:
            break
         
        previous_cost = current_cost
 

        weight_derivative = -(2/n) * (x.transpose() @ (y-y_predicted))
        # Calculating the gradients

        w = w - learning_rate * weight_derivative         
        # Printing the parameters for each 1000th iteration
        print(f"Iteration {i+1}: Cost {current_cost}")
    return w

def main():
     
    X, Y = load_boston(return_X_y=True)
    # X = X.transpose()
    scaler = preprocessing.StandardScaler().fit(X)
    X_scaled = scaler.transform(X)
    X = np.hstack([X_scaled,np.ones((506,1))])
    
    print(X.shape)

    w =  TGCR(X, Y, iterations=3000)

    Y_pred = X @w
 
    # Plotting the regression line
    plt.figure(figsize = (8,6))
    plt.plot( Y, marker='o', color='red')
    plt.plot( Y_pred, marker='+', color='blue')

    plt.ylabel("Y")
    plt.show()
 
     
if __name__=="__main__":
    main()