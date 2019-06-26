#!/usr/bin/env python
# coding: utf-8

# In[428]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[429]:


from scipy.io import loadmat
data = loadmat('ex4data1.mat')
X = data['X']
y = data['y']


# In[430]:


weights = loadmat('ex4weights.mat')
Theta1 = weights['Theta1']
Theta2 = weights['Theta2']


# ## Sigmoid function

# In[431]:


def nn_Sigmoid(x, theta):
    x = np.concatenate((np.array(1).reshape(1,), np.array(x))).reshape((x.shape[0]+1, 1))
    inside = np.array([np.matmul(theta[row,:], x) for row in range(theta.shape[0])]).reshape((theta.shape[0],))
    G = (1 / (1 + np.exp(-inside))).reshape((theta.shape[0],))
    return G


# ## Regularized Cost Function for our NN Model

# In[432]:


from IPython.display import Image
Image(filename = 'neural_net.png')


# $\textbf{Cost function:}$
# $$ J(\theta) = \frac{1}{m}\sum_{i=1}^{m} \sum_{k=1}^{K}
#     \left[-y_k^{(i)}\log((h_\theta(x^{(i)}))_k)
#     -(1-y_k^{(i)})\log(1-(h_\theta(x^{(i)}))_k)
#     \right]+\frac{\lambda}{2m}
#     \left[\sum_{j=1}^{25} \sum_{k=1}^{400}(\Theta_{j,k}^{(1)})^2
#     +\sum_{j=1}^{10} \sum_{k=1}^{25} (\Theta_{j,k}^{(2)})^2
#     \right],$$
# where $m$ is a number of observations, $K$ is a number of predicted classes.

# In[433]:


def CostFunction(Y, y_prob, theta_1, theta_2, par_lambda = 0):
    # computing unregularized cost function
    cost = np.array([[(- Y[obs, category] * np.log(y_prob[obs,category]) -((1 - Y[obs, category]) * np.log(1 - y_prob[obs,category])))
                      for category in range(Y.shape[1])]
                     for obs in range(Y.shape[0])]).mean() * Y.shape[1] #multiplying by a number of classes to get a correct value
    cost = cost + par_lambda / (2*Y.shape[0]) * ((theta_1[:,1:].ravel()**2).sum() + (theta_2[:,1:].ravel()**2).sum())
    
    return cost


# ## First NN model returning the vector of predictions, and the value of cost function

# In[434]:


def NeuralNet_predict(X, y, theta_1, theta_2, par_lambda):
    # Vectorized Sigmoid function
    Sigmoid = lambda X, theta: nn_Sigmoid(X, theta)
    
    # Extracting a 1-0 matrix Y
    Y = np.empty((y.shape[0],len(np.unique(y))))
    for K in np.unique(y):
        col = K - 1
        for i in range(y.shape[0]):
            if y[i,0] == K:
                Y[i,col] = 1
            else: Y[i,col] = 0
    
    # Creating matrix for a hidden layer
    A = np.empty(shape=(X.shape[0], theta_1.shape[0]))    
    # Computing values for a hidden layer
    A = np.array([nn_Sigmoid(X[row,:], theta_1) for row in range(X.shape[0])]).reshape((X.shape[0], theta_1.shape[0]))
    
    # Computing probabilities for an output layer
    y_prob = np.array([nn_Sigmoid(A[row,:], theta_2) for row in range(A.shape[0])]).reshape((X.shape[0], theta_2.shape[0]))
    # Getting predictions from probabilities
    y_pred = np.array([y_prob[row,:].argmax() + 1 for row in range(y_prob.shape[0])]).reshape((y.shape[0], 1))
    
    # Computing the value of our cost function:
    cost = CostFunction(Y, y_prob, theta_1, theta_2, par_lambda)
    return (y_pred, y_prob, cost)


# In[435]:


y_pred, y_prob, cost = NeuralNet_predict(X, y, Theta1, Theta2, par_lambda=1)


# In[436]:


Y = np.empty((y.shape[0],len(np.unique(y))))
for K in np.unique(y):
    col = K - 1
    for i in range(y.shape[0]):
        if y[i,0] == K:
            Y[i,col] = 1
        else: Y[i,col] = 0


# In[437]:


CostFunction(Y, y_prob, Theta1, Theta2, par_lambda=1)


# # Gradually building backpropagation
# ### Sigmoid Gradient

# In[665]:


def sigmoid_gradient(x, theta):
    x = np.concatenate((np.array(1).reshape(1,), np.array(x))).reshape((x.shape[0]+1, 1))
    inside = np.array([np.matmul(theta[row,:], x) for row in range(theta.shape[0])]).reshape((theta.shape[0],))
    G = (1 / (1 + np.exp(-inside))).reshape((theta.shape[0],))
    grad = G * (1-G)
    return grad    


# In[666]:


sigmoid_gradient(X[0,:], Theta1).shape


# ### Adding Random initialization into our model

# In[440]:


def Single_hidden_layer_NN(X, y, n_nodes, epsilon_init = 0.12, par_lambda = 0):
    # Random initialization (i.e., Theta matrices with random weights)
    eps = epsilon_init
    theta_1 = np.random.uniform(low = 0, high = 1, size = (n_nodes, X.shape[1] + 1)) * 2 * eps - eps
    theta_2 = np.random.uniform(0, 1, size = (len(np.unique(y)), n_nodes + 1)) * 2 * eps - eps
    
    # Defining nn_Sigmoid function
    def nn_Sigmoid(x, theta):
        x = np.concatenate((np.array(1).reshape(1,), np.array(x))).reshape((x.shape[0]+1, 1))
        inside = np.array([np.matmul(theta[row,:], x) for row in range(theta.shape[0])]).reshape((theta.shape[0],))
        G = (1 / (1 + np.exp(-inside))).reshape((theta.shape[0],))
        return G
    # Vectorized Sigmoid function
    Sigmoid = lambda X, theta: nn_Sigmoid(X, theta)
    
    # Defining Cost function
    def CostFunction(Y, y_prob, theta_1, theta_2, par_lambda = 0):
        # computing unregularized cost function
        cost = np.array([[(- Y[obs, category] * np.log(y_prob[obs,category]) -((1 - Y[obs, category]) * np.log(1 - y_prob[obs,category])))
                      for category in range(Y.shape[1])]
                     for obs in range(Y.shape[0])]).mean() * Y.shape[1] #multiplying by a number of classes to get a correct value
        cost = cost + par_lambda / (2*Y.shape[0]) * ((theta_1[:,1:].ravel()**2).sum() + (theta_2[:,1:].ravel()**2).sum())
        return cost
    
    # Extracting a 1-0 matrix Y
    Y = np.empty((y.shape[0],len(np.unique(y))))
    for K in np.unique(y):
        col = K - 1
        for i in range(y.shape[0]):
            if y[i,0] == K:
                Y[i,col] = 1
            else: Y[i,col] = 0
    
    # -------------------------------------------Feed-forward propagation-------------------------------------------------
    # Creating matrix for a hidden layer
    A = np.empty(shape=(X.shape[0], theta_1.shape[0]))    
    # Computing values for a hidden layer
    A = np.array([nn_Sigmoid(X[row,:], theta_1) for row in range(X.shape[0])]).reshape((X.shape[0], theta_1.shape[0]))
    
    # Computing probabilities for an output layer
    y_prob = np.array([nn_Sigmoid(A[row,:], theta_2) for row in range(A.shape[0])]).reshape((X.shape[0], theta_2.shape[0]))
    # Getting predictions from probabilities (vector with labels and also 1-0 matrix)
    y_pred = np.array([y_prob[row,:].argmax() + 1 for row in range(y_prob.shape[0])]).reshape((y.shape[0], 1))
    Y_pred = np.empty((y.shape[0], len(np.unique(y))))
    for K in np.unique(y):
        col = K - 1
        for i in range(y.shape[0]):
            if y_pred[i,0] == K:
                Y_pred[i,col] = 1
            else: Y_pred[i,col] = 0
    
    # Computing the value of our cost function:
    cost = CostFunction(Y, y_prob, theta_1, theta_2, par_lambda)
    # -------------------------------------------Feed-forward propagation-------------------------------------------------
    
    # -------------------------------------------Backpropagation----------------------------------------------------------
    ## We need to compute delta_3 & Delta_2. Delta_1 is not computed as we do not want to adjust
    ## feature values.
    
    # Computing error delta_3
    delta_3 = np.array([(y_prob - Y)[:,category].sum() for category in range(Y.shape[1])])
    
    # Computing Delta_2 (for gradient between 2nd and 3rd layer {hidden and ouptut})
    A_extended = np.concatenate((np.array(pd.DataFrame({'bias': np.ones(X.shape[0])})), A), axis = 1)  
   
    Delta_2 = np.empty((len(np.unique(y)), n_nodes + 1)) # initializing a matrix
    for row in (range(X.shape[0])):
        Delta_2 = (Delta_2 + 
                   np.matmul(delta_3.reshape((len(np.unique(y)),1)), (A_extended[row,:].reshape((n_nodes + 1,1))).transpose()))
    
    # Computing Delta_1 (for gradient between 1st and 2nd layer {input and hidden})
    X_extended = np.concatenate((np.array(pd.DataFrame({'bias': np.ones(X.shape[0])})), X), axis = 1)      
    
    Delta_1 = np.empty((n_nodes, X.shape[1] + 1))
    for category in range(len(np.unique(y))):
        for row in range(X.shape[0]):
            Delta_1 = (Delta_1 +
                      np.matmul(Delta_2[category, 1:].reshape((Delta_2.shape[1] - 1,1)),
                                X_extended[row,:].reshape((X.shape[1] + 1,1)).transpose()))
    
    ## Obtaining gradient for a Theta_2 matrix
    # unregularized formula for a bias term
    gradient23_bias = (1/X.shape[0]) * Delta_2[:,0].reshape((Delta_2.shape[0],1))
    # regularized formula for remainig terms
    gradient23_layer = (1/X.shape[0]) * Delta_2[:,1:] + (par_lambda / X.shape[0]) * theta_2[:,1:]
    # combining this together
    gradient23_layer = np.concatenate((gradient23_bias, gradient23_layer), axis = 1)
    
    ## Obtaining gradient for a Theta_1 matrix
    # unregularized formula for a bias term
    gradient12_bias = (1/X.shape[0]) * Delta_1[:,0].reshape((Delta_1.shape[0],1))
    # regularized formula for remainig terms
    gradient12_layer = (1/X.shape[0]) * Delta_1[:,1:] + (par_lambda / X.shape[0]) * theta_1[:,1:]
    # combining this together
    gradient12_layer = np.concatenate((gradient12_bias, gradient12_layer), axis = 1)
                   
    # -------------------------------------------Backpropagation----------------------------------------------------------


# ## Now it is necessary to add while-cycle for implementing Gradient Descent

# In[441]:


def Single_hidden_layer_NN(X, y, n_nodes, max_iter = 500, learning_rate = 0.01, epsilon_init = 0.12, par_lambda = 0):
    ###
    COST = np.zeros(max_iter)
    ###
    # Random initialization (i.e., Theta matrices with random weights)
    eps = epsilon_init
    theta_1 = np.random.uniform(low = 0, high = 1, size = (n_nodes, X.shape[1] + 1)) * 2 * eps - eps
    theta_2 = np.random.uniform(0, 1, size = (len(np.unique(y)), n_nodes + 1)) * 2 * eps - eps
    
    # Defining nn_Sigmoid function
    def nn_Sigmoid(x, theta):
        x = np.concatenate((np.array(1).reshape(1,), np.array(x))).reshape((x.shape[0]+1, 1))
        inside = np.array([np.matmul(theta[row,:], x) for row in range(theta.shape[0])]).reshape((theta.shape[0],))
        G = (1 / (1 + np.exp(-inside))).reshape((theta.shape[0],))
        return G
    # Vectorized Sigmoid function
    Sigmoid = lambda X, theta: nn_Sigmoid(X, theta)
    
    # Defining Cost function
    def CostFunction(Y, y_prob, theta_1, theta_2, par_lambda = 0):
        # computing unregularized cost function
        cost = np.array([[(- Y[obs, category] * np.log(y_prob[obs,category]) -((1 - Y[obs, category]) * np.log(1 - y_prob[obs,category])))
                      for category in range(Y.shape[1])]
                     for obs in range(Y.shape[0])]).mean() * Y.shape[1] #multiplying by a number of classes to get a correct value
        cost = cost + par_lambda / (2*Y.shape[0]) * ((theta_1[:,1:].ravel()**2).sum() + (theta_2[:,1:].ravel()**2).sum())
        return cost
    
    # Extracting a 1-0 matrix Y
    Y = np.empty((y.shape[0],len(np.unique(y))))
    for K in np.unique(y):
        col = K - 1
        for i in range(y.shape[0]):
            if y[i,0] == K:
                Y[i,col] = 1
            else: Y[i,col] = 0
    
    # The core of our algorithm
    n_iter = 0
    while n_iter < max_iter: 
        # -------------------------------------------Feed-forward propagation-------------------------------------------------
        # Creating matrix for a hidden layer
        A = np.empty(shape=(X.shape[0], theta_1.shape[0]))    
        # Computing values for a hidden layer
        A = np.array([nn_Sigmoid(X[row,:], theta_1) for row in range(X.shape[0])]).reshape((X.shape[0], theta_1.shape[0]))

        # Computing probabilities for an output layer
        y_prob = np.array([nn_Sigmoid(A[row,:], theta_2) for row in range(A.shape[0])]).reshape((X.shape[0], theta_2.shape[0]))
        # Getting predictions from probabilities (vector with labels and also 1-0 matrix)
        y_pred = np.array([y_prob[row,:].argmax() + 1 for row in range(y_prob.shape[0])]).reshape((y.shape[0], 1))
        Y_pred = np.empty((y.shape[0], len(np.unique(y))))
        for K in np.unique(y):
            col = K - 1
            for i in range(y.shape[0]):
                if y_pred[i,0] == K:
                    Y_pred[i,col] = 1
                else: Y_pred[i,col] = 0

        # Computing the value of our cost function:
        cost = CostFunction(Y, y_prob, theta_1, theta_2, par_lambda)
        # -------------------------------------------Feed-forward propagation-------------------------------------------------

        # -------------------------------------------Backpropagation----------------------------------------------------------
        ## We need to compute delta_3 & Delta_2. Delta_1 is not computed as we do not want to adjust
        ## feature values.

        # Computing error delta_3
        delta_3 = np.array([(y_prob - Y)[:,category].sum() for category in range(Y.shape[1])])

        # Computing Delta_2 (for gradient between 2nd and 3rd layer {hidden and ouptut})
        A_extended = np.concatenate((np.array(pd.DataFrame({'bias': np.ones(X.shape[0])})), A), axis = 1)  

        Delta_2 = np.empty((len(np.unique(y)), n_nodes + 1)) # initializing a matrix
        for row in (range(X.shape[0])):
            Delta_2 = (Delta_2 + 
                       np.matmul(delta_3.reshape((len(np.unique(y)),1)), (A_extended[row,:].reshape((n_nodes + 1,1))).transpose()))

        # Computing Delta_1 (for gradient between 1st and 2nd layer {input and hidden})
        X_extended = np.concatenate((np.array(pd.DataFrame({'bias': np.ones(X.shape[0])})), X), axis = 1)      

        Delta_1 = np.empty((n_nodes, X.shape[1] + 1))
        for category in range(len(np.unique(y))):
            for row in range(X.shape[0]):
                Delta_1 = (Delta_1 +
                          np.matmul(Delta_2[category, 1:].reshape((Delta_2.shape[1] - 1,1)),
                                    X_extended[row,:].reshape((X.shape[1] + 1,1)).transpose()))

        ## Obtaining gradient for a Theta_2 matrix
        # unregularized formula for a bias term
        gradient23_bias = (1/X.shape[0]) * Delta_2[:,0].reshape((Delta_2.shape[0],1))
        # regularized formula for remainig terms
        gradient23_layer = (1/X.shape[0]) * Delta_2[:,1:] + (par_lambda / X.shape[0]) * theta_2[:,1:]
        # combining this together
        gradient23_layer = np.concatenate((gradient23_bias, gradient23_layer), axis = 1)

        ## Obtaining gradient for a Theta_1 matrix
        # unregularized formula for a bias term
        gradient12_bias = (1/X.shape[0]) * Delta_1[:,0].reshape((Delta_1.shape[0],1))
        # regularized formula for remainig terms
        gradient12_layer = (1/X.shape[0]) * Delta_1[:,1:] + (par_lambda / X.shape[0]) * theta_1[:,1:]
        # combining this together
        gradient12_layer = np.concatenate((gradient12_bias, gradient12_layer), axis = 1)
        # -------------------------------------------Backpropagation----------------------------------------------------------
        
        # -------------------------------------------Learning & adjusting weights---------------------------------------------
        theta_1 = theta_1 - learning_rate * gradient12_layer
        thete_2 = theta_2 - learning_rate * gradient23_layer
        ##
        n_iter = n_iter + 1
        COST[n_iter-1] = cost
    # Returning Theta matrices
    return (theta_1, theta_2, COST)


# In[15]:


THETA1, THETA2, Cost = Single_hidden_layer_NN(X, y, n_nodes = 25, max_iter = 1, learning_rate = 0.1, epsilon_init = 0.12, par_lambda = 0)


# In[16]:


y_pred, cost = NeuralNet_predict(X, y, THETA1, THETA2, par_lambda = 0)


# In[17]:


print('Accuracy is: ', (y == y_pred).sum() / 50, '%')
print('Cost is: ', cost)


# In[18]:


plt.plot(Cost)


# ## Gradient checking

# In[765]:


def Single_hidden_layer_NN(X, y, n_nodes, max_iter = 500, learning_rate = 0.01, epsilon_init = 0.12, par_lambda = 0):
    ###
    pool = list(np.arange(0, 4999))
    COST = []
    ###
    # Random initialization (i.e., Theta matrices with random weights)
    eps = epsilon_init
    theta_1 = np.random.uniform(low = 0, high = 1, size = (n_nodes, X.shape[1] + 1)) * 2 * eps - eps
    theta_2 = np.random.uniform(0, 1, size = (len(np.unique(y)), n_nodes + 1)) * 2 * eps - eps
    
    # Defining nn_Sigmoid function
    def nn_Sigmoid(x, theta):
        x = np.concatenate((np.array(1).reshape(1,), np.array(x))).reshape((x.shape[0]+1, 1))
        inside = np.array([np.matmul(theta[row,:], x) for row in range(theta.shape[0])]).reshape((theta.shape[0],))
        G = (1 / (1 + np.exp(-inside))).reshape((theta.shape[0],))
        return G
    # Vectorized Sigmoid function
    Sigmoid = lambda X, theta: nn_Sigmoid(X, theta)
    
    # Defining Cost function
    def CostFunction(Y, y_prob, theta_1, theta_2, par_lambda = 0):
        # computing unregularized cost function
        cost = np.array([[(- Y[obs, category] * np.log(y_prob[obs,category]) -((1 - Y[obs, category]) * np.log(1 - y_prob[obs,category])))
                      for category in range(Y.shape[1])]
                     for obs in range(Y.shape[0])]).mean() * Y.shape[1] #multiplying by a number of classes to get a correct value
        cost = cost + par_lambda / (2*Y.shape[0]) * ((theta_1[:,1:].ravel()**2).sum() + (theta_2[:,1:].ravel()**2).sum())
        return cost
    
    # Extracting a 1-0 matrix Y
    Y = np.empty((y.shape[0],len(np.unique(y))))
    for K in np.unique(y):
        col = K - 1
        for i in range(y.shape[0]):
            if y[i,0] == K:
                Y[i,col] = 1
            else: Y[i,col] = 0
    
    # The core of our algorithm
    n_iter = 0
    while n_iter < max_iter: 
        # -------------------------------------------Feed-forward propagation-------------------------------------------------
        # Creating matrix for a hidden layer
        A = np.empty(shape=(X.shape[0], theta_1.shape[0]))    
        # Computing values for a hidden layer
        A = np.array([nn_Sigmoid(X[row,:], theta_1) for row in range(X.shape[0])]).reshape((X.shape[0], theta_1.shape[0]))

        # Computing probabilities for an output layer
        y_prob = np.array([nn_Sigmoid(A[row,:], theta_2) for row in range(A.shape[0])]).reshape((X.shape[0], theta_2.shape[0]))
        # Getting predictions from probabilities (vector with labels and also 1-0 matrix)
        y_pred = np.array([y_prob[row,:].argmax() + 1 for row in range(y_prob.shape[0])]).reshape((y.shape[0], 1))
        Y_pred = np.empty((y.shape[0], len(np.unique(y))))
        for K in np.unique(y):
            col = K - 1
            for i in range(y.shape[0]):
                if y_pred[i,0] == K:
                    Y_pred[i,col] = 1
                else: Y_pred[i,col] = 0

        # Computing the value of our cost function:
        cost = CostFunction(Y, y_prob, theta_1, theta_2, par_lambda)
        # -------------------------------------------Feed-forward propagation-------------------------------------------------

        # -------------------------------------------Backpropagation----------------------------------------------------------
        ## We need to compute delta_3 & Delta_2. Delta_1 is not computed as we do not want to adjust
        ## feature values.

        # Computing error delta_3 and delta_2 (they are vectors)
        t = np.random.randint(low = 0, high = 4999 - n_iter)
        obs = pool[t]
        pool.pop(t)
        a_extended = np.concatenate((np.ones((1,)), A[obs,:].reshape((A.shape[1],))))
        x_extended = np.concatenate((np.ones((1,)), X[obs,:].reshape((X.shape[1],))))        
        
        delta_3 = np.array([(y_prob - Y)[obs,category] for category in range(Y.shape[1])]).reshape((Y.shape[1],))
        delta_2 = np.matmul(theta_2.transpose(), delta_3) * (a_extended * (1-a_extended))
        
        # Computing Delta_2 (for gradient between 2nd and 3rd layer {hidden and ouptut})
        Delta_2 = np.empty((len(np.unique(y)), n_nodes + 1)) # initializing a matrix
        Delta_2 = (Delta_2 + 
                   np.matmul(delta_3.reshape((len(np.unique(y)),1)), (a_extended.reshape((n_nodes + 1,1))).transpose()))

        # Computing Delta_1 (for gradient between 1st and 2nd layer {input and hidden})
        Delta_1 = np.empty((n_nodes, X.shape[1] + 1))
        Delta_1 = (Delta_1 + 
                   np.matmul(delta_2[1:].reshape((n_nodes ,1)), (x_extended.reshape((X.shape[1] + 1,1))).transpose()))

        ## Obtaining gradient for a Theta_2 matrix
        # unregularized formula for a bias term
        gradient23_bias = Delta_2[:,0].reshape((Delta_2.shape[0],1))
        # regularized formula for remainig terms
        gradient23_layer = Delta_2[:,1:] + (par_lambda / X.shape[0]) * theta_2[:,1:]
        # combining this together
        gradient23_layer = np.concatenate((gradient23_bias, gradient23_layer), axis = 1)

        ## Obtaining gradient for a Theta_1 matrix
        # unregularized formula for a bias term
        gradient12_bias = Delta_1[:,0].reshape((Delta_1.shape[0],1))
        # regularized formula for remainig terms
        gradient12_layer = Delta_1[:,1:] + (par_lambda / X.shape[0]) * theta_1[:,1:]
        # combining this together
        gradient12_layer = np.concatenate((gradient12_bias, gradient12_layer), axis = 1)
        # -------------------------------------------Backpropagation----------------------------------------------------------
        
        # -------------------------------------------Learning & adjusting weights---------------------------------------------
        theta_1 = theta_1 - learning_rate * gradient12_layer
        theta_2 = theta_2 - learning_rate * gradient23_layer
        ##
        n_iter = n_iter + 1
        COST.append(cost)
        print('[', n_iter, '] ', cost, -(COST[n_iter-2] - COST[n_iter-1]) / COST[n_iter-2] * 100, '%')
    # Returning Theta matrices
    return ((COST, obs, theta_1, theta_2, gradient12_layer, gradient23_layer, y_prob))


# In[859]:


cost, obs, THETA1, THETA2, G1, G2, y_prob = Single_hidden_layer_NN(X, y, n_nodes = 50)


# In[860]:


sns.heatmap(y_prob)


# In[867]:


plt.plot(cost)[20:]


# In[862]:


y_pred, y_prob, cost = NeuralNet_predict(X, y, THETA1, THETA2, par_lambda=0)
print('Accuracy is: ', (y_pred == y).sum()/50, '%')
print('Unique values of y are: ', np.unique(y_pred))
print('Value of a cost function is: ', cost)


# In[858]:


def Single_hidden_layer_NN(X, y, n_nodes, n_batch = 20, n_runs = 1, max_iter = 10000, base_learning_rate = 0.005, epsilon_init = 0.12, par_lambda = 0):
    lowest_cost = 100
    # Defining nn_Sigmoid function
    def nn_Sigmoid(x, theta):
        x = np.concatenate((np.array(1).reshape(1,), np.array(x))).reshape((x.shape[0]+1, 1))
        inside = np.array([np.matmul(theta[row,:], x) for row in range(theta.shape[0])]).reshape((theta.shape[0],))
        G = (1 / (1 + np.exp(-inside))).reshape((theta.shape[0],))
        return G
    # Vectorized Sigmoid function
    Sigmoid = lambda X, theta: nn_Sigmoid(X, theta)
    
    # Defining Cost function
    def CostFunction(Y, y_prob, theta_1, theta_2, par_lambda = 0):
        # computing unregularized cost function
        cost = np.array([[(- Y[obs, category] * np.log(y_prob[obs,category]) -((1 - Y[obs, category]) * np.log(1 - y_prob[obs,category])))
                      for category in range(Y.shape[1])]
                     for obs in range(Y.shape[0])]).mean() * Y.shape[1] #multiplying by a number of classes to get a correct value
        cost = cost + par_lambda / (2*Y.shape[0]) * ((theta_1[:,1:].ravel()**2).sum() + (theta_2[:,1:].ravel()**2).sum())
        return cost
    
    # Extracting a 1-0 matrix Y
    Y = np.empty((y.shape[0],len(np.unique(y))))
    for K in np.unique(y):
        col = K - 1
        for i in range(y.shape[0]):
            if y[i,0] == K:
                Y[i,col] = 1
            else: Y[i,col] = 0
    
    # The core of our algorithm
    for run in range(n_runs):
        n_iter = 0
        # Random initialization (i.e., Theta matrices with random weights)
        eps = epsilon_init
        theta_1 = np.random.uniform(low = 0, high = 1, size = (n_nodes, X.shape[1] + 1)) * 2 * eps - eps
        theta_2 = np.random.uniform(0, 1, size = (len(np.unique(y)), n_nodes + 1)) * 2 * eps - eps
        ###
        pool = list(np.arange(0, 4999))
        COST = []
        ###    
        while n_iter < max_iter: 
            # -------------------------------------------Feed-forward propagation-------------------------------------------------
            # Creating matrix for a hidden layer
            A = np.empty(shape=(X.shape[0], theta_1.shape[0]))    
            # Computing values for a hidden layer
            A = np.array([nn_Sigmoid(X[row,:], theta_1) for row in range(X.shape[0])]).reshape((X.shape[0], theta_1.shape[0]))

            # Computing probabilities for an output layer
            y_prob = np.array([nn_Sigmoid(A[row,:], theta_2) for row in range(A.shape[0])]).reshape((X.shape[0], theta_2.shape[0]))
            # Getting predictions from probabilities (vector with labels and also 1-0 matrix)
            y_pred = np.array([y_prob[row,:].argmax() + 1 for row in range(y_prob.shape[0])]).reshape((y.shape[0], 1))
            Y_pred = np.empty((y.shape[0], len(np.unique(y))))
            for K in np.unique(y):
                col = K - 1
                for i in range(y.shape[0]):
                    if y_pred[i,0] == K:
                        Y_pred[i,col] = 1
                    else: Y_pred[i,col] = 0

            # Computing the value of our cost function:
            cost = CostFunction(Y, y_prob, theta_1, theta_2, par_lambda)
            # -------------------------------------------Feed-forward propagation-------------------------------------------------

            # -------------------------------------------Backpropagation----------------------------------------------------------
            ## We need to compute delta_3 & Delta_2. Delta_1 is not computed as we do not want to adjust
            ## feature values.

            # Computing error delta_3 and delta_2 (they are vectors)
            #t = -np.sort(np.random.choice(-np.arange(0, 4999 - n_iter * 10), size = 20))
            t = np.sort(np.random.choice(np.arange(0, 4999), size = n_batch))
            Delta_2 = np.empty((len(np.unique(y)), n_nodes + 1)) # initializing a matrix
            Delta_1 = np.empty((n_nodes, X.shape[1] + 1))
            for i in range(n_batch):
                obs = pool[t[i]]
                a_extended = np.concatenate((np.ones((1,)), A[obs,:].reshape((A.shape[1],))))
                x_extended = np.concatenate((np.ones((1,)), X[obs,:].reshape((X.shape[1],))))        

                delta_3 = np.array([(y_prob - Y)[obs,category] for category in range(Y.shape[1])]).reshape((Y.shape[1],))
                delta_2 = np.matmul(theta_2.transpose(), delta_3) * (a_extended * (1-a_extended))

                # Computing Delta_2 (for gradient between 2nd and 3rd layer {hidden and ouptut})
                Delta_2 = (Delta_2 + 
                            np.matmul(delta_3.reshape((len(np.unique(y)),1)), (a_extended.reshape((n_nodes + 1,1))).transpose()))

                # Computing Delta_1 (for gradient between 1st and 2nd layer {input and hidden})
                Delta_1 = (Delta_1 + 
                           np.matmul(delta_2[1:].reshape((n_nodes ,1)), (x_extended.reshape((X.shape[1] + 1,1))).transpose()))

                ## Obtaining gradient for a Theta_2 matrix
                # unregularized formula for a bias term
                gradient23_bias = (1/n_batch) * Delta_2[:,0].reshape((Delta_2.shape[0],1))
                # regularized formula for remainig terms
                gradient23_layer = (1/n_batch) * Delta_2[:,1:] + (par_lambda / n_batch) * theta_2[:,1:]
                # combining this together
                gradient23_layer = np.concatenate((gradient23_bias, gradient23_layer), axis = 1)

                ## Obtaining gradient for a Theta_1 matrix
                # unregularized formula for a bias term
                gradient12_bias = (1/n_batch) * Delta_1[:,0].reshape((Delta_1.shape[0],1))
                # regularized formula for remainig terms
                gradient12_layer = (1/n_batch) * Delta_1[:,1:] + (par_lambda / n_batch) * theta_1[:,1:]
                # combining this together
                gradient12_layer = np.concatenate((gradient12_bias, gradient12_layer), axis = 1)
            #for i in range(n_batch):
                #pool.pop(t[i])
            # -------------------------------------------Backpropagation----------------------------------------------------------

            # -------------------------------------------Learning & adjusting weights---------------------------------------------
            learning_rate = base_learning_rate + 0.1 / (np.sqrt(n_iter+1))
            theta_1 = theta_1 - learning_rate * gradient12_layer
            theta_2 = theta_2 - learning_rate * gradient23_layer
            ##
            n_iter = n_iter + 1
            COST.append(cost)
            print('Run: ', run+1, 'iter: ', '[', n_iter, '] ', cost, -(COST[n_iter-2] - COST[n_iter-1]) / COST[n_iter-2] * 100, '%')
        if cost < lowest_cost:
            best_theta_1 = theta_1
            best_theta_2 = theta_2
            lowest_COST = COST
            best_y_prob = y_prob
    # Returning Theta matrices
    return ((COST, obs, best_theta_1, best_theta_2, gradient12_layer, gradient23_layer, best_y_prob))


# In[ ]:




