import numpy as np
import scipy.io
from nn import *
import matplotlib.pyplot as plt
import ipdb

def get_accuracy_graph(accuracy_per_epoch_list,max_iters):
    epoch_count = np.arange(1,max_iters+1)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(epoch_count, accuracy_per_epoch_list)
    ax.set_xlabel('No. of Epochs')
    ax.set_ylabel('Accuracy')
    ax.set_title('Accuracy VS No. of Epochs')
    plt.show()

def get_loss_graph(total_loss_per_epoch_list,max_iters):
    epoch_count = np.arange(1,max_iters+1)
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(epoch_count, total_loss_per_epoch_list)
    ax.set_xlabel('No. of Epochs')
    ax.set_ylabel('Cross entropy Loss')
    ax.set_title('Cross Entropy loss VS No. of Epochs')
    plt.show()

train_data = scipy.io.loadmat('../data/nist36_train.mat')
valid_data = scipy.io.loadmat('../data/nist36_valid.mat')
test_data = scipy.io.loadmat('../data/nist36_test.mat')

train_x, train_y = train_data['train_data'], train_data['train_labels']
valid_x, valid_y = valid_data['valid_data'], valid_data['valid_labels']
test_x, test_y = test_data['test_data'], test_data['test_labels']

max_iters = 60
# pick a batch size, learning rate
batch_size = train_x.shape[0]//1000
learning_rate = .003          
'''
Tried learning rates: 
                        0.1 --------------> 22.58%
                        0.01 -------------> 72.8%
                        0.0015 ------------> 73.19%
                        0.0025 ------------> 74.52%
                        0.005 ------------> 74.58%
                        0.001 ------------> 72.02%
'''
hidden_size = 64

#Training Data
batches = get_random_batches(train_x,train_y,batch_size)
batch_num = len(batches)

#Validation Data
batches_validation = get_random_batches(valid_x,valid_y,valid_x.shape[0])

params = {}

# initialize layers here
initialize_weights(1024,64,params,'layer1')
initialize_weights(64,36,params,'output')
assert(params['Wlayer1'].shape == (1024,64))
assert(params['blayer1'].shape == (64,))

weights_before_network_training = params['Wlayer1']
# ipdb.set_trace()

# with default settings, you should get loss < 150 and accuracy > 80%
accuracy_per_epoch_list = []
total_loss_per_epoch_list = []
validation_accuracy_per_epoch_list = []
validation_loss_per_epoch_list = []
for itr in range(max_iters):
    total_loss = 0
    total_acc = 0
    for xb,yb in batches:
        # training loop can be exactly the same as q2!
        # forward
        h1 = forward(xb,params,'layer1')
        probs = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        loss, acc = compute_loss_and_acc(yb, probs)
        total_loss+=loss
        total_acc+=acc

        # backward
        delta1 = probs
        delta1[np.arange(probs.shape[0]),np.argmax(yb, axis=1)] -= 1
        delta2 = backwards(delta1,params,'output',linear_deriv)
        backwards(delta2,params,'layer1',sigmoid_deriv)

        # apply gradient

        for layer in ['output', 'layer1']:
            params['W' + layer] -= learning_rate * params['grad_W' + layer]
            params['b' + layer] -= learning_rate * params['grad_b' + layer]

    total_acc/=len(batches) 
    accuracy_per_epoch_list.append(total_acc)
    total_loss_per_epoch_list.append(total_loss)

    for xb_valid,yb_valid in batches_validation:
        # forward
        h1 = forward(xb_valid,params,'layer1')
        probs_valid = forward(h1,params,'output',softmax)

        # loss
        # be sure to add loss and accuracy to epoch totals 
        validation_loss, valid_acc = compute_loss_and_acc(yb_valid, probs_valid)

    validation_accuracy_per_epoch_list.append(valid_acc)
    validation_loss_per_epoch_list.append(validation_loss)
       
    if itr % 2 == 0:
        print("itr: {:02d} \t loss: {:.2f} \t acc : {:.2f}".format(itr,total_loss,total_acc))    #This is the average accuracy for all the batches combined but for a specific iteration count

'''
See if the method I am calculating accuracy by is correct or not?  What do they mean by total_acc
Why do we need the validation data in loss/accuracy VS no. of iterations. Validation data should only involve forward pass.
Even if we need it, where do we run the forward pass for the validation data?
'''

get_accuracy_graph(accuracy_per_epoch_list,max_iters)
get_loss_graph(total_loss_per_epoch_list,max_iters)
get_accuracy_graph(validation_accuracy_per_epoch_list,max_iters)
get_loss_graph(validation_loss_per_epoch_list,max_iters)

weights_after_network_training = params['Wlayer1']

# run on validation set and report accuracy! should be above 75%
# batches = get_random_batches(valid_x,valid_y,valid_x.shape[0])

# for xb,yb in batches:
#     # forward
#     h1 = forward(xb,params,'layer1')
#     probs = forward(h1,params,'output',softmax)

#     # loss
#     # be sure to add loss and accuracy to epoch totals 
#     validation_loss, valid_acc = compute_loss_and_acc(yb, probs)

# print('Validation accuracy: ',valid_acc)

# if False: # view the data
#     for crop in xb:
#         import matplotlib.pyplot as plt
#         plt.imshow(crop.reshape(32,32).T)
#         plt.show()

batches = get_random_batches(test_x,test_y,test_x.shape[0])
#batch_num = len(batches)
#print("No. of batches considered for validation data are: ",batch_num)

for xb,yb in batches:
    # forward
    h1 = forward(xb,params,'layer1')
    probs = forward(h1,params,'output',softmax)

    # loss
    # be sure to add loss and accuracy to epoch totals 
    test_loss, test_acc = compute_loss_and_acc(yb, probs)

print('Test Accuracy: ',test_acc)

if False: # view the data
    for crop in xb:
        import matplotlib.pyplot as plt
        plt.imshow(crop.reshape(32,32).T)
        plt.show()

# import pickle
# saved_params = {k:v for k,v in params.items() if '_' not in k}
# with open('q3_weights.pickle', 'wb') as handle:
#     pickle.dump(saved_params, handle, protocol=pickle.HIGHEST_PROTOCOL)

# Q3.1.3
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

fig = plt.figure(1, (4., 4.))
grid = ImageGrid(fig, 111,  # similar to subplot(111)
                 nrows_ncols=(1, 2),  # creates 1x2 grid of axes
                 axes_pad=0.1,  # pad between axes in inch.
                 )

grid[0].imshow(weights_before_network_training)
grid[1].imshow(weights_after_network_training)
plt.show()


# Q3.1.4
confusion_matrix = np.zeros((train_y.shape[1],train_y.shape[1]))
result = np.argmax(yb, axis=1)  #This is the set of true labels -------------------------------->This is temporary. Kindly change
predicted_outputs = np.argmax(probs, axis=1) #This is the set of predicted labels -------------------------------->This is temporary. Kindly change
for i,elt in enumerate(result):
    confusion_matrix[predicted_outputs[i]][result[i]]+=1

import string
plt.imshow(confusion_matrix,interpolation='nearest')
plt.grid(True)
plt.xticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.yticks(np.arange(36),string.ascii_uppercase[:26] + ''.join([str(_) for _ in range(10)]))
plt.show()