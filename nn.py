import numpy as np
from util import *
# do not include any more libraries here!
# do not put any code outside of functions!
#import ipdb
# Q 2.1
# initialize b to 0 vector
# b should be a 1D array, not a 2D array with a singleton dimension
# we will do XW + b. 
# X be [Examples, Dimensions]
def initialize_weights(in_size,out_size,params,name=''):
    W, b = None, None

    U = (6/(in_size+out_size))**0.5 #confirm value

    W = np.random.uniform(-U,U,(in_size,out_size))  

    b = np.zeros(out_size) #Made in the shape of a row vector as of now, so that it can be added to WX
    
    params['W' + name] = W
    params['b' + name] = b

# Q 2.2.1
# x is a matrix
# a sigmoid activation function
def sigmoid(x):
    res = None
    func = lambda t: 1/(1+np.exp(-t))
    vfunc = np.vectorize(func)
    res = vfunc(x)
    return res

# Q 2.2.2
def forward(X,params,name='',activation=sigmoid):
    """
    Do a forward pass

    Keyword arguments:
    X -- input vector [Examples x D]
    params -- a dictionary containing parameters
    name -- name of the layer
    activation -- the activation function (default is sigmoid)
    """
    pre_act, post_act = None, None
    # get the layer parameters
    W = params['W' + name]
    b = params['b' + name]

    # your code here
    
    pre_act = np.matmul(X,W) + b      

    post_act = activation(pre_act)  

    # store the pre-activation and post-activation values
    # these will be important in backprop
    params['cache_' + name] = (X, pre_act, post_act)

    return post_act

# Q 2.2.2 
# x is [examples,classes]
# softmax should be done for each row
def softmax(x):
    '''
    Find max of each of the rows (one row is one data point). Subtract this max from each of elements in a row (The stability trick).
    Find the exponential of each element in the array. Find softmax by dividing the sum of the elements in the row by each element . 
    res is softmax of each class. The array shape is [examples,classes]
    '''
    res = None
    x = x - np.amax(x, axis=1)[:,None]  
    x = np.exp(x)
    res = x / np.sum(x, axis=1)[:,None]  

    return res

# Q 2.2.3
# compute total loss and accuracy
# y is size [examples,classes]
# probs is size [examples,classes]
def compute_loss_and_acc(y, probs):
    '''
    To find the accuracy find the label of the data from the one hot vector encoding by finding the row-wise argmax in the y.
    Also find the argmax of the probs. Compare this with the vector obtained above.

    To find the loss, use the argmax of y obtained above. Use a loop over that array & multiply y with the log of the value at the corresponding index in probs. 
    Keep summing these up and take a minus at the end.
    '''
    loss, acc = None, None

    true_outputs = np.argmax(y, axis=1)            #Stores the indices of the true output
    predicted_outputs = np.argmax(probs, axis=1) #Stores the indices of the estimated output
    #print("===================================================",true_outputs.shape,"\n",predicted_outputs.shape,"\n")
    assert true_outputs.shape[0] == predicted_outputs.shape[0], "Unequal length of output vectors of y and probs"
    correct_predictions = np.count_nonzero(true_outputs==predicted_outputs)
    acc = correct_predictions/true_outputs.shape[0] 

    loss = 0
    for i,elt in enumerate(true_outputs):
        loss-= np.log(probs[i,elt])
   
    return loss, acc 

# we give this to you
# because you proved it
# it's a function of post_act
def sigmoid_deriv(post_act):
    res = post_act*(1.0-post_act)
    return res

def backwards(delta,params,name='',activation_deriv=sigmoid_deriv):
    """
    Do a backwards pass

    Keyword arguments:
    delta -- errors to backprop
    params -- a dictionary containing parameters
    name -- name of the layer
    activation_deriv -- the derivative of the activation_func
    """
    grad_X, grad_W, grad_b = None, None, None
    # everything you may need for this layer
    W = params['W' + name]
    b = params['b' + name]
    X, pre_act, post_act = params['cache_' + name]
    # your code here
    # do the derivative through activation first
    # then compute the derivative W,b, and X

    #think of 'a' as the linear combo of x's and 'z' as the post-activation of 'a'
    # if(activation_deriv == sigmoid_deriv):  #Handle other cases as well
    #     dz_da = sigmoid_deriv(post_act)

    dz_da = delta*activation_deriv(post_act)

    grad_W = np.matmul(X.T,dz_da) 
    #da_db is simply identity
    grad_X = np.matmul(dz_da,W.T)  
    grad_b = np.full(b.shape, np.sum(dz_da, axis=0))

    # store the gradients
    params['grad_W' + name] = grad_W
    params['grad_b' + name] = grad_b
    return grad_X

# Q 2.4
# split x and y into random batches
# return a list of [(batch1_x,batch1_y)...]
def get_random_batches(x,y,batch_size):
    assert batch_size <= x.shape[0], "Batch size is larger than the total number of data points"
    batches = []
    no_of_batches_wanted = x.shape[0]//batch_size
    for i in range(no_of_batches_wanted):
        idx = np.random.choice(x.shape[0], batch_size, replace=False)
        x_sample = x[idx,:]
        y_sample = y[idx,:]
        batches.append((x_sample,y_sample))
    
    return batches
