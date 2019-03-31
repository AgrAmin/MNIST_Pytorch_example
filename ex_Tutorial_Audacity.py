'''
import torch
from torch.autograd import Variable
import torch.nn.functional as F
#import torch.nn as nn #
#import torchvision.datasets as dsets #
#import torchvision.transforms as transforms #
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
#import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing #for normalization
'''
#https://www.youtube.com/watch?v=zFA8Cm13Xmk
#https://towardsdatascience.com/how-to-train-an-image-classifier-in-pytorch-and-use-it-to-perform-basic-inference-on-single-images-99465a1e9bf5
#https://www.youtube.com/watch?v=zN49HdDxHi8 #data loder issue and more stuff worth saving or pringting.
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html#sphx-glr-download-beginner-blitz-cifar10-tutorial-py

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

#Listing the labels training & test set
mypathTrain=r"C:\Users\Asus\Desktop\CH\colormix\seedr\course dl\practicex\zero_to_deep_learning_video\data\spoken_numbers_pcm\imgTrain"
trainlabel = [f for f in listdir(mypathTrain) if isfile(join(mypathTrain, f))]

mypathTest=r"C:\Users\Asus\Desktop\CH\colormix\seedr\course dl\practicex\zero_to_deep_learning_video\data\spoken_numbers_pcm\imgTest"
testlabel = [g for g in listdir(mypathTest) if isfile(join(mypathTest, g))]

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

# Transforms for the training, validation, and testing sets
#training_transforms, testing_transforms = processing_functions.data_transforms()

# Load the datasets with ImageFolder
#training_dataset, testing_dataset = processing_functions.load_datasets(mypathTrain, transform, mypathTest, transform)

testing_dataset=datasets.ImageFolder(root=mypathTrain,transform=transform)
training_dataset=datasets.ImageFolder(root=mypathTest,transform=transform)

train_loader=torch.utils.data.DataLoader(training_dataset,batch_size=8, shuffle= True)
test_loader=torch.utils.data.DataLoader(training_dataset,batch_size=8, shuffle= True)

print(type(test_loader))
print(type(testing_dataset))
class_names= testing_dataset.classes
print(class_names)
#torchvision.datasets.folder.ImageFolder

# Build and train your network
# Transfer Learning
device = torch.device("cuda" if torch.cuda.is_available()
                                  else "cpu")
model = models.vgg16(pretrained=True)
print(model)

##### load and show some images for fun #####
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# get some random training images
dataiter = iter(train_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images))
# print labels
print(' '.join('%5s' % class_names[labels[j]] for j in range(8)))  #range(batch_size)
####End test loading images ### note the number of image showed = batch size ###########

### CNN ###
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 *117*157, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #print(x.size(3))
        x = x.view(-1,16*117*157)#(len(x[0]),len(x))#(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


net = Net()

####### loss fucntion and optimizer ###
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
########"
'''
############### freezing the hidden layers ##############
for param in model.parameters():
    param.requires_grad = False

model.fc = nn.Sequential(nn.Linear(2048, 512),
                         nn.ReLU(),
                         nn.Dropout(0.2),
                         nn.Linear(512, 10),
                         nn.LogSoftmax(dim=1))
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.fc.parameters(), lr=0.003)
model.to(device)


############# #################

################Training ###################
epochs = 1
steps = 0
running_loss = 0
print_every = 10
train_losses, test_losses = [], []

for epoch in range(epochs):
    for inputs, labels in train_loader:
        steps += 1
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        logps = model.forward(inputs)
        loss = criterion(logps, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0
            model.eval()
            with torch.no_grad():
                for inputs, labels in test_loader:
                    inputs, labels = inputs.to(device),
                    labels.to(device)
                logps = model.forward(inputs)
                batch_loss = criterion(logps, labels)
                test_loss += batch_loss.item()

                ps = torch.exp(logps)
                top_p, top_class = ps.topk(1, dim=1)
                #equals =
                top_class == labels.view(*top_class.shape)
            #accuracy +=
        torch.mean(equals.type(torch.FloatTensor)).item()
    train_losses.append(running_loss / len(train_loader))
    test_losses.append(test_loss / len(test_loader))
    print(f"Epoch {epoch+1}/{epochs}.. "
          f"Train loss: {running_loss/print_every:.3f}.. "
          f"Test loss: {test_loss/len(test_loader):.3f}.. "
          f"Test accuracy: {accuracy/len(test_loader):.3f}")
    running_loss = 0
    model.train()
torch.save(model, 'aerialmodel.pth')
###############
'''

for epoch in range(15):  # loop over the dataset multiple times #2 -> 14% accuracy; 10->64%; 25->100%

    running_loss = 0.0
    for i, data in enumerate(train_loader, 0):
        # get the inputs
        inputs, labels = data
        #print(torch.Tensor.size(inputs))

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()
        if i % 2000 == 1999:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, running_loss / 2000))
            running_loss = 0.0

print('Finished Training')

##########
#############Test the Model ###########
dataiter = iter(test_loader)
images, labels = dataiter.next()

# print images
imshow(torchvision.utils.make_grid(images))
print('GroundTruth: ', ' '.join('%5s' % class_names[labels[j]] for j in range(8)))  #range(batch_size)
outputs = net(images)
####################

'''The outputs are energies for the 10 classes. The higher the energy for a class,
 the more the network thinks that the image is of the particular class.
  So, let’s get the index of the highest energy:'''

_, predicted = torch.max(outputs, 1)

print('Predicted: ', ' '.join('%5s' % class_names[predicted[j]]
                              for j in range(8)))
##########accuracy for the whole dataset ##########
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = net(images)
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
        outputs = net(images)
        _, predicted = torch.max(outputs, 1)
        c = (predicted == labels).squeeze()
        for i in range(4):
            label = labels[i]
            class_correct[label] += c[i].item()
            class_total[label] += 1


for i in range(10):
    print('Accuracy of %5s : %2d %%' % (
        class_names[i], 100 * class_correct[i] / class_total[i]))
