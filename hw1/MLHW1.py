
# coding: utf-8

# #Machine Learning Homework 1: Ridge Regression and SGD
# Due Friday, Feb 6 2015
# 
# ---
# ##1 Introduction
# ---

# In[1]:

#Imports and load data
import pandas as pd
import logging
import numpy as np
import sys
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
import os
get_ipython().magic(u'matplotlib inline')
import timeit
from IPython.display import Image

#Loading the dataset
print('loading the dataset')

df = pd.read_csv('../hw1/hw1-data.csv', delimiter=',')
X = df.values[:,:-1]
y = df.values[:,-1]

print('Split into Train and Test')
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=100, random_state=10)


# ---
# ##2 Linear Regression
# ---
# ###2.1 Feature Normalization
# Modify function `feature_normalization` to normalize all the features to [0,1].  (Can you use numpy's "broadcasting" here?)
# 
# >Numpy's broadcasting would be used here if train and test were different sizes.  We are broadcasting training arrays on the test set.  

# In[99]:

def feature_normalization(train, test):
    """Rescale the data so that each feature in the training set is in
    the interval [0,1], and apply the same transformations to the test
    set, using the statistics computed on the training set.
    
    Args:
        train - training set, a 2D numpy array of size (num_instances, num_features)
        test  - test set, a 2D numpy array of size (num_instances, num_features)
    Returns:
        train_normalized - training set after normalization
        test_normalized  - test set after normalization

    """    

    train_range = np.ptp(train, axis=0)
    train_range[train_range==0]=1
    train_min = np.min(train, axis=0)
    train_norm = (train - train_min)/train_range
    test_norm = (test - train_min)/train_range

    return train_norm,test_norm

print("Scaling all to [0, 1]")
X_train, X_test = feature_normalization(X_train, X_test)    
X_train = np.hstack((X_train, np.ones((X_train.shape[0], 1)))) #Add bias term
X_test = np.hstack((X_test, np.ones((X_test.shape[0], 1)))) #Add bias term


# ---
# ###2.2 Gradient Descent Setup
# 1. Write the objective function $J(\theta)$ as a matrix/vector expression, without using an explicit summation sign.  
# >$J(\theta)=\frac{1}{2m}(X\theta - y)^T(X\theta - y)$
# 
# 2. Write down an expression for the gradient of $J$.
# >$\nabla J(\theta) = \frac{1}{m}(X\theta - y)^TX$
# 
# 3. Use the gradient to write down an approximate expression for $J(\theta + \eta \Delta)-J(\theta)$.
# >$J(\theta + \eta \Delta)-J(\theta) \approx \nabla J(\theta) \Delta \eta $
# 
# 4. Write down the expression for updating $\theta$ in the gradient descent algorithm.  Let $\eta$ be the step size.  
# >$\theta_{i+1} = \theta_i - \eta * \nabla J(\theta)$
# 
# 5. Modify the function `compute_square_loss`, to compute $J(\theta)$ for a given $\theta$.
# >See next cell
# 
# 6. Modify the function `compute_square_loss_gradient`, to compute $\nabla J(\theta)$

# In[4]:

########################################
####Q2.2a: The square loss function

def compute_square_loss(X, y, theta, lambda_reg=0):
    """
    Given a set of X, y, theta, compute the square loss for predicting y with X*theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D array of size (num_features)
    
    Returns:
        loss - the square loss, scalar
    """
    loss = 0 #initialize the square_loss
    m=X.shape[0]
    yhat = np.dot(X,theta)
    loss = 1.0/2/m * np.dot(yhat-y,yhat-y) + lambda_reg*np.dot(theta,theta)
    return loss


########################################
###Q2.2b: compute the gradient of square loss function
def compute_square_loss_gradient(X, y, theta):
    """
    Compute gradient of the square loss (as defined in compute_square_loss), at the point theta.
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    m=X.shape[0]
    yhat = np.dot(X,theta)
    grad = 1.0/m * np.dot(yhat - y,X)
    return grad


# ---
# ###2.3 Gradient Checker
# 1. Complete the function `grad_checker` according to the documentation given. 

# In[5]:

def grad_checker(X, y, theta, epsilon=0.01, tolerance=1e-4): 
    """Implement Gradient Checker
    Check that the function compute_square_loss_gradient returns the
    correct gradient for the given X, y, and theta.

    Let d be the number of features. Here we numerically estimate the
    gradient by approximating the directional derivative in each of
    the d coordinate directions: 
    (e_1 = (1,0,0,...,0), e_2 = (0,1,0,...,0), ..., e_d = (0,...,0,1) 

    The approximation for the directional derivative of J at the point
    theta in the direction e_i is given by: 
    ( J(theta + epsilon * e_i) - J(theta - epsilon * e_i) ) / (2*epsilon).

    We then look at the Euclidean distance between the gradient
    computed using this approximation and the gradient computed by
    compute_square_loss_gradient(X, y, theta).  If the Euclidean
    distance exceeds tolerance, we say the gradient is incorrect.

    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        epsilon - the epsilon used in approximation
        tolerance - the tolerance error
    
    Return:
        A boolean value indicate whether the gradient is correct or not

    """
    true_gradient = compute_square_loss_gradient(X, y, theta) #the true gradient
    num_features = theta.shape[0]
    approx_grad = np.zeros(num_features) #Initialize the gradient we approximate
    
    for i in range(0,num_features):
        e = np.zeros(num_features)
        e[i]=1
    
        approx_grad[i] = (compute_square_loss(X,y,theta + epsilon * e) - compute_square_loss(X,y,theta - epsilon*e))*1.0 / (2.0*epsilon)
    
    dist = np.linalg.norm(true_gradient - approx_grad)
    correct_grad = dist<tolerance
    assert correct_grad, "Gradient bad: dist %s is greater than tolerance %s" % (dist,tolerance)
    return correct_grad


# ---
# ###2.4 Batch Gradient Descent
# 1. Complete `batch_gradient_descent`
# >See next cell
# 
# 2. Starting with a step-size of 0.1 (not a bad one to start with), try various different fixed step sizes to see which converges most quickly. Plot the value of the objective function as a function of the number of steps. Briefly summarize your findings.
# >See next cell

# In[6]:

####################################
####Q2.4a: Batch Gradient Descent
def batch_grad_descent(X, y, alpha=0.1, num_iter=1000, check_grad=False):
    """
    In this question you will implement batch gradient descent to
    minimize the square loss objective
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        num_iter - number of iterations to run 
        check_grad - a boolean value indicating whether checking the gradient when updating
        
    Returns:
        theta_hist - store the the history of parameter vector in iteration, 2D numpy array of size (num_iter+1, num_features) 
                    for instance, theta in iteration 0 should be theta_hist[0], theta in ieration (num_iter) is theta_hist[-1]
        loss_hist - the history of objective function vector, 1D numpy array of size (num_iter+1) 
    """
    (num_instances, num_features) = X.shape
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #initialize loss_hist
    theta = np.ones(num_features) #initialize theta
        
    for i in range(num_iter):
        theta_hist[i] = theta
        loss_hist[i] = compute_square_loss(X,y,theta)
        if check_grad:
            grad_check = grad_checker(X,y,theta)
            print "grad_check:",grad_check
        grad = compute_square_loss_gradient(X,y,theta)
        theta = theta - alpha*grad
       
    theta_hist[num_iter] = theta
    loss_hist[num_iter] = compute_square_loss(X,y,theta)
    return loss_hist,theta_hist

####Q2.4b: Plot convergence at various step sizes
def plot_step_convergence(X,y,num_iter=1000):
    """
    Plots instances of batch_grad_descent at various step_sizes (alphas)
    """
    step_sizes = [0.001, 0.01, 0.05, 0.1, 0.101]
    for step_size in step_sizes:
        loss_hist,_ = batch_grad_descent(X,y,alpha=step_size, num_iter=num_iter)
        plt.plot(range(num_iter+1),loss_hist, label=step_size)
    
    plt.xlabel('Steps')
    plt.ylabel('Loss')
    plt.yscale('log')    
    plt.title('Convergence Rates by Step Size')
    plt.legend()
    plt.show()
    
plot_step_convergence(X_train,y_train)


# ---
# ###2.5 Ridge Regression (i.e. Linear Regression with $L_2$ regularization)
# 1. Compute the gradient of $J(\theta)$ and write down the expression for updating $\theta$ in the gradient descent algorithm.
# >$\nabla J(\theta) = \frac{1}{m}(X\theta - y)^TX + 2\lambda \theta ^T$
# 
# 2. Implement `compute regularized square loss gradient`.
# > See next cell.
# 
# 3. Implement `regularized grad descent`.
# > See next cell.
# 
# 4. Explain why making $B$ large decreases the effective regularization on the bias term, and how we can make that regularization as weak as we like (though not zero).
# > The bias term represents
# $\hat{y} = B*\theta_B$ when $X=0$. So a larger $B$ means smaller $\theta_B$, before regularization; and a smaller penalty for weight in the bias term. 
# 
# 5. Start with $B = 1$. Choosing a reasonable step-size, find the $\theta _\lambda^∗$ that minimizes $J(\theta)$ for a range of λ and plot both the training loss and the validation loss as a function of λ. (Note that this is just the square loss, not including the regularization term.) You should initially try λ over several orders of magnitude to find an appropriate range (e.g . $λ ∈ 􏰀\{10^{−2}, 10^{−1}, 1, 10,100\}$􏰁. You may want to have $log(λ)$ on the $x$-axis rather than λ. Once you have found the interesting range for λ, repeat the fits with different values for $B$, and plot the results on the same graph. For this dataset, does regularizing the bias help, hurt, or make no significant difference?
# >See next cell
# 
# 6. Estimate the average time it takes on your computer to compute a single gradient step.
# >I ran a test on the regularized gradient descent function, and it took approximately 69 microsends to run 1000 steps, which translates to 69 nanoseconds per step. See code below
# 
# 7. What $\theta$ would you select for deployment and why?
# > I believe this question is asking for $\lambda$.  I found the minimum square loss to be 1.4, at $\lambda = 0.01$
# 

# In[94]:

###################################################
###Q2.5a: Compute the gradient of Regularized Batch Gradient Descent
def compute_regularized_square_loss_gradient(X, y, theta, lambda_reg):
    """
    Compute the gradient of L2-regularized square loss function given X, y and theta
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        theta - the parameter vector, 1D numpy array of size (num_features)
        lambda_reg - the regularization coefficient
    
    Returns:
        grad - gradient vector, 1D numpy array of size (num_features)
    """
    m=X.shape[0]
    yhat = np.dot(X,theta)
    grad = 1.0/m * np.dot((yhat - y),X) + 2.0*lambda_reg*theta
    return grad


###################################################
###Q2.5b: Batch Gradient Descent with regularization term
def regularized_grad_descent(X, y, alpha=0.1, lambda_reg=1, num_iter=1000):
    """
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - step size in gradient descent
        lambda_reg - the regularization coefficient
        numIter - number of iterations to run 
        
    Returns:
        theta_hist - the history of parameter vector, 2D numpy array of size (num_iter+1, num_features) 
        loss_hist - the history of regularized loss value, 1D numpy array
    """
    (num_instances, num_features) = X.shape
    theta = np.ones(num_features) #Initialize theta
    theta_hist = np.zeros((num_iter+1, num_features))  #Initialize theta_hist
    loss_hist = np.zeros(num_iter+1) #Initialize loss_hist
        
    for i in range(num_iter+1):
        theta_hist[i] = theta
        loss_hist[i] = compute_square_loss(X,y,theta,lambda_reg=lambda_reg)
        grad = compute_regularized_square_loss_gradient(X,y,theta,lambda_reg)
        #Make gradient a unit vector
        theta = theta - alpha*grad/np.linalg.norm(grad)
    
    assert loss_hist[0]>0, "Loss history[0] is still zero"
    assert theta_hist[0,0]>0, "Theta_hist[0] is is still zero"
    return loss_hist,theta_hist
    
#############################################
##Q2.5c: Visualization of Regularized Batch Gradient Descent

def plot_regularized_grad(X_tr,y_tr,X_val,y_val, alpha=0.1, num_iter=1000):
    """
    Args:
        X_tr - the feature vector, 2D numpy array of size (num_instances, num_features)
        y_tr - the label vector, 1D numpy array of size (num_instances)
        X_val - the feature vector from test data
        y_val - the label vector from test data
        alpha - step size in gradient descent
        numIter - number of iterations to run 
        
    Returns:
        Plot
        X-axis: log(lambda_reg)
        Y-axis: square_loss (training and test)  
    """
    biases = [1,5,10,20]
    colors = ['c','g','y','r']
    lambda_exponents = np.arange(-4,3,0.5)
    lambda_regs = map(lambda x: 10**x, lambda_exponents)
    
    #initialize square loss
    training_loss = np.zeros(len(lambda_regs))
    test_loss = np.zeros(len(lambda_regs))    
    
    #initialize plot
    fig = plt.figure()
    ax = plt.subplot(111)
    
    first_run=True
    for j,bias in enumerate(biases):
        #adjust bias term
        X_tr[:,-1] = bias
        
        for i,lambda_reg in enumerate(lambda_regs):
            loss_hist,theta_hist = regularized_grad_descent(X_tr,y_tr, alpha,lambda_reg, num_iter)
            training_loss[i] = loss_hist[-1]
            test_loss[i] = compute_square_loss(X_val,y_val,theta_hist[-1])
            
            #Record new low-loss mark
            if first_run or test_loss[i]<min_test_loss[0]:
                min_test_loss=[test_loss[i],bias,lambda_reg]
            first_run=False                
        
        ax.plot(lambda_regs,training_loss,'--%s' %colors[j],label = "training B=%s" % bias)
        ax.plot(lambda_regs,test_loss,'-%s' %colors[j],label = "validation B=%s" % bias)
            
    # Shrink current axis by 20%
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    ax.set_title('Loss, Varying Bias Term "B" and Lambda')
    ax.set_xscale('log')
    ax.set_xlabel('Regularization term Lambda')
    ax.set_ylabel('Square Loss') 
    plt.show()
    print "Minimum loss is %f, found at Bias=%d and Lambda=%f" %(round(min_test_loss[0],1),min_test_loss[1],min_test_loss[2])


# In[95]:

plot_regularized_grad(X_train,y_train,X_test,y_test)


# In[87]:

#############################################
##Q2.5.6: Estimate the average time it takes on your computer 
##to compute a single gradient step.

def timeme(func,*args,**kwargs):
    """
    Timer wrapper.  Runs a given function, with arguments,
    100 times and displays the average time per run.  
    """
    def wrapper(func, *args, **kwargs):
        def wrapped():
            return func(*args, **kwargs)
        return wrapped
    wrapped = wrapper(func,*args,**kwargs)
    run_time = float(timeit.timeit(wrapped, number=100))*10
    print "Avg time to run %s after 100 trials: %i µs per trial" %(func,run_time)
    
timeme(regularized_grad_descent,X_train, y_train)


# ---
# ###2.6 Stochastic Gradient Descent
# 1. Write down the update rule for $\theta$ in SGD.
# >Stochastic gradient at point $i$ is given by $$\nabla J_i(\theta) = (\vec{x_i}^T\theta - y_i)\vec{x_i} + 2\lambda \theta^T$$
# where $\vec{x_i}$ is the feature vector for instance $i$ and $y_i$ is a scalar
# 
# 2. Implement stochastic_grad_descent
# > See next cell
# 
# 3. Use SGD to find $θλ^∗$ that minimizes the ridge regression objective for the $λ$ and $B$ that you selected in the previous problem. Try several different fixed step sizes, as well as step sizes that decrease with the step number according to the following schedules: $η = \frac{1}{t}$ and $η = \frac{1}{\sqrt{t}}$ Plot the value of the objective function (or the log of the objective function if that is more clear) as a function of epoch (or step number) for each of the approaches to step size. How do the results compare? (Note: In this case we are investigating the convergence rate of the optimization algorithm, thus we’re interested in the value of the objective function, which includes the regularization term.)
# > See next cell
# 
# 4. Estimate the amount of time it takes on your computer for a single epoch of SGD.
# > The test below showed that 100 epochs takes 550µs, or 5.5µs per epoch.  
# 
# 5. Comparing SGD and gradient descent, if your goal is to minimize the total number of epochs (for SGD) or steps (for batch gradient descent), which would you choose? If your goal were to minimize the total time, which would you choose?
# > Gradient descent converges in 1000 steps; 69 nanoseconds per step, that's 69µs.  SGD converges in fewer than 10 epochs; at 5.5µs per epoch, that's less than 50µs.  SGD is fewer steps and less time.  

# In[113]:

#############################################
###Q2.6a: Stochastic Gradient Descent    
def compute_stochastic_gradient(X_i,y_i,theta, lambda_reg):
    yhat = np.dot(X_i,theta)
    grad = (yhat - y_i)*X_i + 2.0*lambda_reg*theta
    return grad

def stochastic_grad_descent(X,y,alpha=0.1, lambda_reg=1, num_iter=100):
    """
    In this question you will implement stochastic gradient descent with a regularization term
    
    Args:
        X - the feature vector, 2D numpy array of size (num_instances, num_features)
        y - the label vector, 1D numpy array of size (num_instances)
        alpha - string or float. step size in gradient descent
                NOTE: In SGD, it's not always a good idea to use a fixed step size. Usually it's set to 1/sqrt(t) or 1/t
                if alpha is a float, then the step size in every iteration is alpha.
                if alpha == "1/sqrt(t)", alpha = 1/sqrt(t)
                if alpha == "1/t", alpha = 1/t
        lambda_reg - the regularization coefficient
        num_iter - number of epochs (i.e number of times) to go through the whole training set
    
    Returns:
        theta_hist - the history of parameter vector, 3D numpy array of size (num_iter, num_instances, num_features) 
        loss hist - the history of regularized loss function vector, 2D numpy array of size(num_iter, num_instances)
    """
    num_instances, num_features = X.shape
    theta = np.ones(num_features) #Initialize theta
    
    theta_hist = np.zeros((num_iter, num_instances, num_features))  #Initialize theta_hist
    loss_hist = np.zeros((num_iter, num_instances)) #Initialize loss_hist
    step_size = 0.01
    t=1
    index_order = range(num_instances)
    
    for epoch in range(num_iter):
        #maybe random shuffle here
        #np.random.shuffle(index_order)
        for i, rand_idx in enumerate(index_order):            
            #options for alpha are float, 1/t, or 1/sqrt(t)
            if alpha=="1/t":
                step_size=1.0/float(t)
            elif alpha=="1/sqrt(t)":
                step_size=1.0/np.sqrt(t)
            else:
                step_size=alpha
            
            theta_hist[epoch,i] = theta
            loss_hist[epoch,i] = compute_square_loss(X,y,theta,lambda_reg)
            grad = compute_stochastic_gradient(X[rand_idx,:],y[rand_idx],theta,lambda_reg)
            theta = theta - step_size*grad
            t=t+1

    return loss_hist,theta_hist
    

################################################
###Q2.6b Visualization that compares the convergence speed of batch
###and stochastic gradient descent for various approaches to step_size
##X-axis: Step number (for gradient descent) or Epoch (for SGD)
##Y-axis: log(objective_function_value)
def plot_stochastic(X,y,alpha=0.1, lambda_reg=1, num_iter=20):
    num_instances, num_features = X.shape
    alphas = [0.01,0.05,"1/t","1/sqrt(t)"]

    #initialize plot
    fig = plt.figure()
    ax = plt.subplot(111)
       
    
    for i,alpha in enumerate(alphas):        
        loss_hist,_ = stochastic_grad_descent(X,y,alpha,lambda_reg, num_iter)
        #plot the last instance from each iteration
        ax.plot(range(num_iter),loss_hist[:,-1], label="alpha=%s" % alpha)
    
    # Shrink current axis by 20%
    box = ax.get_position()
    #Position legend
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    
    ax.set_title('Rates of convergence for various Alphas')
    ax.set_yscale('log')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Square Loss') 
    plt.show()


# In[114]:

plot_stochastic(X_train,y_train)


# In[106]:

#################
##Q2.6.4
timeme(stochastic_grad_descent,X_train,y_train)


# ---
# ##3 Risk Minimization
# 1. Show that for the square loss $\ell(\hat{y}, y) = \frac{1}{2}(y − \hat{y})^2$, the Bayes decision function is a $f_∗(x) = \mathbb{E} [Y | X = x]$. [Hint: Consider constructing $f_∗ (x)$, one $x$ at a time.]
# >See image below:

# In[4]:

Image(filename='files/image.png')


# In[ ]:



