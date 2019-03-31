
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
#from torch import nn
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models


from os import listdir
from os.path import isfile, join

mypathTrain=r"C:\data\DataNumTr"
trainlabel = [f for f in listdir(mypathTrain) if isfile(join(mypathTrain, f))]

#transforma = transforms.Compose(
 #   [transforms.ToTensor(),
  #   transforms.Normalize((0.1307), (0.3081))]) #transform (mean) (std)
transforma = transforms.ToTensor()

train_dataset=datasets.MNIST(root=mypathTrain,train= True,download=False,transform=transforma)
test_dataset=datasets.MNIST(root=mypathTrain,train=False,download=False,transform=transforma)

train_loader=torch.utils.data.DataLoader(train_dataset,batch_size=64, shuffle= True)
test_loader=torch.utils.data.DataLoader(test_dataset,batch_size=64, shuffle= True)

print((test_dataset.classes))

##### load and show some images for fun #####
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dattaiter = iter(test_loader)
images, labels = dattaiter.next() #******

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
        self.hidden1= nn.Linear(28*28,128) #
        self.hidden2= nn.Linear(128,64)
        #output layer
        self.output= nn.Linear(64,10)#

    def forward(self,x):
        # flatten image
        x = x.view(-1, 28 * 28)
        x = F.relu(self.hidden1(x))
        x = F.relu(self.hidden2(x))
        x = F.softmax(self.output(x),dim=1)
        return x

classifierv2= nn.Sequential(nn.Linear(28*28,128),
                            nn.ReLU(),
                            nn.Linear(128,64),
                            nn.ReLU(),
                            nn.Linear(64,10),
                            nn.LogSoftmax())
#####################################"
#net = classifierv1()#if you use this remove line 69 or remove resahping of input tensor each time
net = classifierv2 #same as v1 just cleaner code

####### loss fucntion and optimizer ###
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.01, momentum=0.9)


for epoch in range(15):  # loop over the dataset multiple times #2 -> 14% accuracy; 10->64%; 25->100%

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



print('Finished Training')

##########accuracy for the whole dataset ##########
correct = 0
total = 0
with torch.no_grad():
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
