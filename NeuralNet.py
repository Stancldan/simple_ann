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