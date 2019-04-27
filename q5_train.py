import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import pdb
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim
import time


transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])   #How do I get the mean and the standard deviation. Also change this for 1 channel

train_set = torchvision.datasets.EMNIST(root='./emnist_data', split= 'balanced',train=True, download=True, transform=transform)
test_set = torchvision.datasets.EMNIST(root='./emnist_data',split= 'balanced', train=False, download=True, transform=transform)


# img, lab = train_set.__getitem__(0)
# print(img.shape)

# test_set = torchvision.datasets.CIFAR10(root='./cifardata', train=False, download=True, transform=transform)

classes = ['0','1','2','3','4','5','6','7','8','9',
            'A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            'a','b','d','e','f','g','h','n','q','r','t']

#Training
n_training_samples = 20000                                                            #Tune this number
train_sampler = SubsetRandomSampler(np.arange(n_training_samples, dtype=np.int64))


n_val_samples = 5000
val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

n_test_samples = 5000
test_sampler = SubsetRandomSampler(np.arange(n_test_samples, dtype=np.int64))

#Validation (Understand)
# n_val_samples = 5000
# val_sampler = SubsetRandomSampler(np.arange(n_training_samples, n_training_samples + n_val_samples, dtype=np.int64))

class SimpleCNN(torch.nn.Module):
    
    #Batch shape for input x is (3, 32, 32)
    
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        #Input channels = 3, output channels = 32
        self.conv1 = torch.nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1) #Padding is one, so no reduction in size for a 3X3 filter during convolution
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = torch.nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)    #Tuning the output layer number!! I've taken 36 as of now!!
        self.conv2_drop = torch.nn.Dropout2d()                                          #When and where to use is a design decision!!
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv3 = torch.nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)    #Tuning the output layer number!! I've taken 36 as of now!!                                        #When and where to use is a design decision!!
        #self.pool3 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        #4608 input features, 64 output features (see sizing flow below)
        #self.conv4 = torch.nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1) 
        self.fc1 = torch.nn.Linear(7 * 7 * 128, 500)                                 #Tuning the output layer number!! I've taken 100 as of now!!
        
        #100 input features, 47 output features for our 47 defined classes
        self.fc2 = torch.nn.Linear(500, 47)
        
    def forward(self, x):
        # pdb.set_trace() 
        x = F.relu(self.pool1(self.conv1(x)))                        #Is there another argument which is passed here?

        x = F.relu(self.pool2(self.conv2(x)))       #Linear dropout or 2D drop out. Pros and cons?

        x = F.elu((self.conv3(x)))
        # print(x.size)

        x = x.view(-1, 7 * 7 * 128)

        x = F.relu(self.fc1(x))     #Should I dropout here again?           #Relu here or not?

        x = self.fc2(x)                                                     #Where do we put softmax?

        #x = F.relu(self.conv1(x))
        
        #Size changes from (18, 32, 32) to (18, 16, 16)
        #x = self.pool(x)
        
        #Reshape data to input to the input layer of the neural net
        #Size changes from (18, 16, 16) to (1, 4608)
        #Recall that the -1 infers this dimension from the other given dimension
        # x = x.view(-1, 18 * 16 *16)
        
        #Computes the activation of the first fully connected layer
        #Size changes from (1, 4608) to (1, 64)
        # x = F.relu(self.fc1(x))
        
        #Computes the second fully connected layer (activation applied later)
        #Size changes from (1, 64) to (1, 10)

        return(x)


def get_train_loader(batch_size):
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, sampler=train_sampler, num_workers=2)
    return(train_loader)

test_loader = torch.utils.data.DataLoader(test_set, batch_size=4, sampler=test_sampler, num_workers=2)
val_loader = torch.utils.data.DataLoader(train_set, batch_size=128, sampler=val_sampler, num_workers=2)

def createLossAndOptimizer(net, learning_rate=0.0005):
    
    #Loss function
    loss = torch.nn.CrossEntropyLoss()
    
    #Optimizer
    optimizer = optim.Adam(net.parameters(), lr=learning_rate)
    
    return(loss, optimizer)

def trainNet(net, batch_size, n_epochs, learning_rate):
    
    #Print all of the hyperparameters of the training iteration:
    print("===== HYPERPARAMETERS =====")
    print("batch_size=", batch_size)
    print("epochs=", n_epochs)
    print("learning_rate=", learning_rate)
    print("=" * 30)
    
    #Get training data
    train_loader = get_train_loader(batch_size)
    n_batches = len(train_loader)
    
    #Create our loss and optimizer functions
    loss, optimizer = createLossAndOptimizer(net, learning_rate)
    
    #Time for printing
    training_start_time = time.time()
    
    #Loop for n_epochs
    for epoch in range(n_epochs):
        
        running_loss = 0.0
        print_every = n_batches // 10
        start_time = time.time()
        total_train_loss = 0
        
        for i, data in enumerate(train_loader, 0):
            
            #Get inputs
            inputs, labels = data
            
            #Wrap them in a Variable object
            inputs, labels = Variable(inputs), Variable(labels)         #What is a variable object?
            
            #Set the parameter gradients to zero
            optimizer.zero_grad()
            
            #Forward pass, backward pass, optimize
            outputs = net(inputs)
            loss_size = loss(outputs, labels)
            loss_size.backward()                                        #??
            optimizer.step()                                            #??

            # print(loss_size)
            
            #Print statistics
            # running_loss += loss_size.data[0]          
            # total_train_loss += loss_size.data[0]
            running_loss += loss_size.item()         
            total_train_loss += loss_size.item() 

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(outputs.data, 1)
            correct = (predicted == labels).sum().item()

            
            # #Print every 10th batch of an epoch
            # if (i + 1) % (print_every + 1) == 0:
            #     print("Epoch {}, {:d}% \t train_loss: {:.2f} took: {:.2f}s".format(
            #             epoch+1, int(100 * (i+1) / n_batches), running_loss / print_every, time.time() - start_time))

            #     #Reset running loss and time
            #     running_loss = 0.0
            #     start_time = time.time()

            if (i + 1) % 100 == 0:
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                      .format(epoch + 1, n_epochs, i + 1, n_batches, loss_size.item(),
                              (correct / total) * 100))
            
        #At the end of the epoch, do a pass on the validation set
        total_val_loss = 0
        for inputs, labels in val_loader:
            
            #Wrap tensors in Variables
            inputs, labels = Variable(inputs), Variable(labels)
            
            #Forward pass
            val_outputs = net(inputs)
            val_loss_size = loss(val_outputs, labels)
            total_val_loss += val_loss_size.item()

            # Track the accuracy
            total = labels.size(0)
            _, predicted = torch.max(val_outputs.data, 1)
            correct = (predicted == labels).sum().item()
            
        print("Validation loss = {:.2f}".format(total_val_loss / len(val_loader)))
        print("Validation accuracy is ",(correct / total) * 100," %")

    #torch.save(CNN.state_dict(), "cnn_weights")   
    print("Training finished, took {:.2f}s".format(time.time() - training_start_time))


if __name__ == "__main__":

    CNN = SimpleCNN()
    trainNet(CNN, batch_size=32, n_epochs=5, learning_rate=0.0001)




