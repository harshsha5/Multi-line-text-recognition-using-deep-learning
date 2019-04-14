import numpy as np
import ipdb
from util import *
import nn

if __name__ == '__main__':
	params = {}

	#To test ff
	X = np.array([[1,2,3],[4,5,6]])
	nn.initialize_weights(X.shape[1],2,params)
	f_post = nn.forward(X,params)

	#To test sigmoid
	# arr = np.array([1,3,5])
	# res = nn.sigmoid(arr)

	#To test softmax
	# X = np.array([[1,2,3],[4,5,8]])
	# res = nn.softmax(X)

	#To test compute_loss_and_acc
	# y = np.array([[1,0,0],[0,0,1]])
	# probs = np.array([[0.5,0.2,0.3],[0.4,0.3,0.3]])
	# loss,acc = nn.compute_loss_and_acc(y,probs)
	# print(loss)
	# print(acc)

	#To test get_random_batches
	# x = np.array([[1,2,3],[4,5,8],[51,21,3],[14,25,48],[11,12,23],[44,55,68],[71,82,39],[40,58,87]])
	# y = np.array([[1],[1],[0],[2],[4],[3],[0],[0]])
	# batch_size = 5
	# batches = nn.get_random_batches(x,y,batch_size)
	
	ipdb.set_trace()