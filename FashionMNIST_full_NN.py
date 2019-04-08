import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision

import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from os import listdir
from os.path import isfile, join

from torch.utils.data.sampler import SubsetRandomSampler


mypathTrain=r"C:\Users\Asus\Desktop\CH\colormix\seedr\course dl\practicex\zero_to_deep_learning_video\data\spoken_numbers_pcm\safefash"
trainlabel = [f for f in listdir(mypathTrain) if isfile(join(mypathTrain, f))]

mypathTest=r"C:\Users\Asus\Desktop\CH\colormix\seedr\course dl\practicex\zero_to_deep_learning_video\data\spoken_numbers_pcm\safe\DataNumT"
testlabel = [g for g in listdir(mypathTest) if isfile(join(mypathTest, g))]



#transforma = transforms.Compose(
 #   [transforms.ToTensor(),
  #   transforms.Normalize((0.1307), (0.3081))]) #transform (mean) (std)
transforma = transforms.ToTensor()

train_dataset=datasets.FashionMNIST(root=mypathTrain,train= True,download=False,transform=transforma)
test_dataset=datasets.FashionMNIST(root=mypathTrain,train=False,download=False,transform=transforma)

# obtain training indices that will be used for validation
valid_size=0.2
num_train = len(train_dataset)
indices = list(range(num_train)) 
np.random.shuffle(indices)
split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]
# define samplers for obtaining training and validation batches
train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64,
                                           sampler=train_sampler) #sampler option is mutually exclusive with shuffle
valid_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, sampler=valid_sampler)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=True)

print((test_dataset.classes))

##### load and show some images for fun #####
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dattaiter = iter(test_loader)
images, labels = dattaiter.next() #***************************************

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % train_dataset.classes[labels[j]] for j in range(64)))  #range(batch_size)
####End test loading images ### note the number of image showed = batch size ###########


print(torch.tensor(images.shape))
BigT=images
BigTen=BigT.reshape(64,28*28)
print(BigTen.shape)



class classifierv1 (nn.Module):
    def __init__(self):
        super().__init__()
        #hidden NN
        self.hidden1= nn.Linear(28*28,128*2) #-+-+- 256 <-> 128
        self.hidden2= nn.Linear(128*2,64*2)
        self.hidden3 = nn.Linear(64 * 2, 64)
        #self.hidden=.......
        #output layer
        self.output= nn.Linear(64,10)#-+-+- 256 <-> 64
        #activation function
        #dropout
        self.droupout= nn.Dropout(p=0.2)

    def forward(self,x):
        # flatten image
        #x = x.view(-1, 28 * 28)
        x = self.droupout(F.relu(self.hidden1(x)))
        x = self.droupout(F.relu(self.hidden2(x)))
        x = self.droupout(F.relu(self.hidden3(x)))
        x = F.softmax(self.output(x),dim=1)
        return x

classifierv2= nn.Sequential(nn.Linear(28*28,128*2),
                            nn.ReLU(),
                            nn.Linear(128*2,64*2),
                            nn.ReLU(),
                            nn.Linear(64*2,64),
                            nn.ReLU(),
                            nn.Linear(64,10),
                            nn.LogSoftmax())
#####################################"
net = classifierv1()
#net = classifierv2

####### loss fucntion and optimizer ###
criterion = nn.CrossEntropyLoss()
#optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
optimizer = optim.Adam(net.parameters(), lr=0.001)

R_Loss=[]
R_Lossx=[]
valid_loss_min=100

for epoch in range(25):  # loop over the dataset multiple times

    net.train()
    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        #print(torch.Tensor.size(inputs))
        inputss=inputs.reshape(-1,28*28)
        #print(torch.Tensor.size(inputss))
        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputss)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    else:
        R_Loss.append(running_loss / len(train_loader))
        print(f'Training Loss - Runing Loss: {R_Loss[-1]}')
        running_lossx = 0.0

        net.eval()
        for cnt, datax in enumerate(valid_loader, 0):
            # get the inputs
            inputsx, labelsx = datax
            # print(torch.Tensor.size(inputs))
            inputssx = inputsx.reshape(-1, 28 * 28)
            # print(torch.Tensor.size(inputss))
            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputsx = net(inputssx)
            lossx = criterion(outputsx, labelsx)
            running_lossx += lossx.item()
        else:
            R_Lossx.append(running_lossx / len(valid_loader))
            print(f'Validation Loss - Runing Loss: {R_Lossx[-1]}')
            # save model if validation loss has decreased
            if R_Lossx[-1] <= valid_loss_min:
                print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
                    valid_loss_min,
                    R_Lossx[-1]))
                torch.save(net.state_dict(), 'net.pt')
                valid_loss_min = R_Lossx[-1]

print('Finished Training')
plt.plot(R_Loss,'b')
plt.plot(R_Lossx,'r')
plt.show()


#############Test the Model ###########

##########accuracy for the whole dataset ##########
correct = 0
total = 0
net.eval() #strop the dropout process so the program run faster
with torch.no_grad(): #turn off the Gradient because its not needed and the program will run slightly faster
    for data in test_loader:
        images, labels = data
        imagess = images.reshape(-1, 28 * 28)
        outputs = net(imagess)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the 10000 test images: %d %%' % (
    100 * correct / total))

############ performance evaluation of every class #########
class_correct = list(0. for i in range(10))
class_total = list(0. for i in range(10))
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        imagess = images.reshape(-1, 28 * 28)
        outputs = net(imagess)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        train_dataset.classes[i], 100 * class_correct[i] / class_total[i]))
