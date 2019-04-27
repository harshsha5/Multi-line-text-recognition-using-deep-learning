import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
from run_q4 import *
from q4 import *
from q5_train import SimpleCNN

# transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]) 
# #transform_for_test = transforms.Normalize((0.1307,), (0.3081,))
# train_set = torchvision.datasets.EMNIST(root='./emnist_data', split= 'balanced',train=True, download=True, transform=transform)
# n_training_samples = 20000                                                            #Tune this number
# train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))
# def get_train_loader(batch_size):
#     train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
#     return(train_loader)

def display_input(X):
    X_new = X.numpy()
    for i in range(X_new.shape[0]):
        array = np.reshape(X_new[i,:],(28,28))
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.imshow(array,cmap='Greys',  interpolation='nearest')
        plt.show()

def create_input_dataset(bboxes,bw,input_size = 32):
    X = None
    for i in range (bboxes.shape[0]):    
        minr, minc, maxr, maxc = bboxes[i,:]
        array = bw[minr:maxr,minc:maxc]             #See if this is supposed to be minc:maxc...
        array = square_and_pad_pytorch(array,input_size)

        flattened_array = array.flatten()                   #Check!

        if(i==0):
            X = flattened_array
        else:
            X = np.vstack((X,flattened_array))

    return X

def square_and_pad_pytorch(array,input_size):
    height,width = array.shape
    extra_pad = 30
    if(height>width):
        deficit_pad = int((height-width)/2)
        padding=((extra_pad,extra_pad),(deficit_pad + extra_pad,deficit_pad+extra_pad))
    else:
        deficit_pad = int((width-height)/2)
        padding=((deficit_pad + extra_pad,deficit_pad + extra_pad),(extra_pad,extra_pad))

    array = np.pad(array, padding, mode='constant', constant_values=1)
    #print("Array post padding: ",array.shape)
    array = skimage.transform.resize(array, (input_size, input_size),anti_aliasing=True)                            #See if anti aliasing is required
    #array = skimage.morphology.erosion(array,skimage.morphology.square(3))
    #array = skimage.morphology.erosion(array,skimage.morphology.square(3))                              #Remove square and try. Haikus is good
    # array = skimage.morphology.erosion(array)
    array = skimage.morphology.erosion(array)
    return array



if __name__ == "__main__":
    possible_classification_outputs = np.asarray(['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'a','b','d','e','f','g','h','n','q','r','t'])

    for img in os.listdir('../images'):
        # if(True):
        if(True):
            # train_loader = get_train_loader(batch_size = 20000)

            im1 = skimage.img_as_float(skimage.io.imread(os.path.join('../images',img)))
            bboxes, bw = findLetters(im1)

            for i in range (bboxes.shape[0]):    
                minr, minc, maxr, maxc = bboxes[i,:]
                rect = matplotlib.patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                        fill=False, edgecolor='red', linewidth=2)
                plt.gca().add_patch(rect)

            bboxes,indices_of_new_line,indices_of_spaces = order_bounding_boxes_as_rows(bboxes,bw)
            bboxes = bboxes.astype(int)
            X_numpy = create_input_dataset(bboxes,bw,28)

            # for i in range(X_numpy.shape[0]):
            #     arr = np.reshape(X_numpy[i,:],(28,28))
            #     fig, ax = plt.subplots(figsize=(10, 6))
            #     ax.imshow(arr,cmap='Greys',  interpolation='nearest')
            #     plt.show()
               # # X_numpy[i,:] = (np.rot90(arr)).flatten()



            # fig, ax = plt.subplots(figsize=(10, 6))
            # ax.imshow(np.reshape(X_numpy[1,:], (28,28)))
            # plt.show()
            X = torch.from_numpy(X_numpy).float()
            X = torch.reshape(X, (X_numpy.shape[0], 1,28,28))
            #X = (X - .1307)/.3081
            
            CNN = SimpleCNN()
            CNN.load_state_dict(torch.load("cnn_weights"))
            CNN.eval()
            #with torch.no_grad():

            X = 1-X
            X[X>0.25] = X[X>0.25] ** 0.25
            X[X<=0.25] = X[X<=0.25] ** 3
            X = (X - .1307)/.3081

            X = 1-X
            #display_input(X)

            outputs = CNN(X)
            _, predicted = torch.max(outputs.data, 1)
            #ipdb.set_trace()
            # print(predicted)
            predicted_outputs = predicted.numpy()

            # with torch.no_grad():
            #     count = 0
            #     for inputs, labels in train_loader:
            #         print(count)
            #         outputs = CNN(inputs)                                           

            #         # Track the accuracy
            #         total = labels.size(0)
            #         _, predicted = torch.max(outputs.data, 1)
            #         correct = (predicted == labels).sum().item()
            #         print((correct / total) * 100)

            get_output_text(predicted_outputs,possible_classification_outputs,indices_of_new_line,indices_of_spaces)
            print("====================================================================================================")
