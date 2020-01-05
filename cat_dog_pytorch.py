"""
Transfer Learning
we will use pre-trained networks to solved challenging problems in computer vision.
here we will use Neural Net trained on ImageNet dataset .

ImageNet is a massive dataset with over 1 million labeled images in 1000 categories.
It's used to train deep neural networks using an architecture called convolutional layers. 

Using a pre-trained network on images not in the training set is called transfer learning. 
Here we'll use transfer learning to train a network that can classify our cat and dog photos with near perfect accuracy.
"""

# Loading Image Data
"""
We'll be using a dataset of cat and dog photos available from Kaggle.
link : https://www.kaggle.com/c/dogs-vs-cats
"""

"""
--------------------------------------------------------------------------------------------------------
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt
import pandas as pd
import os
import torch
from torchvision import datasets, transforms

import helper
"""
--------------------------------------------------------------------------------------------------------
"""
filenames = os.listdir("./catdog_kaggle/train")
categories = []
for filename in filenames:
    category = filename.split('.')[0]
    if category == 'dog':
        categories.append(1)
    else:
        categories.append(0)

df = pd.DataFrame({
    'filename': filenames,
    'category': categories
})

"""
--------------------------------------------------------------------------------------------------------
"""
data_dir = './Cat_Dog_data/train/'

#Compose transforms here
transform = transforms.Compose([transforms.Resize(255),transforms.CenterCrop(224),transforms.ToTensor()])
#Create the ImageFolder
dataset = datasets.ImageFolder(data_dir,transform=transform)
#Use the ImageFolder dataset to create the DataLoader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle = True)

"""
--------------------------------------------------------------------------------------------------------
The easiest way to load image data is with datasets.ImageFolder from torchvision 
documantation link : https://pytorch.org/docs/master/torchvision/datasets.html#imagefolder

dataset = datasets.ImageFolder('path/to/data', transform=transform)

where 'path/to/data' is the file path to the data directory and transform is a list of processing steps 
built with the transforms module from torchvision. ImageFolder expects the files and directories to be constructed like so:

root/dog/xxx.png
root/dog/xxy.png
root/dog/xxz.png

root/cat/123.png
root/cat/nsdf3.png
root/cat/asd932_.png

where each class has it's own directory (cat and dog) for the images. 
The images are then labeled with the class taken from the directory name. 
So here, the image 123.png would be loaded with the class label cat. 
You can download the dataset already structured like this from here. 
(link :https://s3.amazonaws.com/content.udacity-data.com/nd089/Cat_Dog_data.zip)
I've also split it into a training set and test set.
--------------------------------------------------------------------------------------------------------
Transforms

  When you load in the data with ImageFolder, you'll need to define some transforms. 
  For example, the images are different sizes but we'll need them to all be the same size for training. 
  You can either resize them with transforms.Resize() or crop with transforms.CenterCrop(), transforms.RandomResizedCrop(), etc.
  We'll also need to convert the images to PyTorch tensors with transforms.ToTensor(). 
  Typically you'll combine these transforms into a pipeline with transforms.Compose(),
  which accepts a list of transforms and runs them in sequence. It looks something like this to scale,
  then crop, then convert to a tensor:

  transform = transforms.Compose([transforms.Resize(255),
                                   transforms.CenterCrop(224),
                                   transforms.ToTensor()])

  There are plenty of transforms available, you can read through the documentation. 
  (link : http://pytorch.org/docs/master/torchvision/transforms.html)
--------------------------------------------------------------------------------------------------------
Data Loaders

  With the ImageFolder loaded, you have to pass it to a DataLoader. 
  The DataLoader takes a dataset (such as you would get from ImageFolder) and returns batches of images
  and the corresponding labels. You can set various parameters like the batch size and if the data is shuffled after each epoch.

  dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)

  Here dataloader is a generator. To get data out of it, you need to loop through it or convert it to an iterator and call next().

  # Looping through it, get a batch on each loop 
  for images, labels in dataloader:
      pass

  # Get one batch
  images, labels = next(iter(dataloader))
--------------------------------------------------------------------------------------------------------
"""

# Run this to test your data loader
images, labels = next(iter(dataloader))
helper.imshow(images[0], normalize=False)

"""
--------------------------------------------------------------------------------------------------------
Data Augmentation

A common strategy for training neural networks is to introduce randomness in the input data itself. 
For example, you can randomly rotate, mirror, scale, and/or crop your images during training.
This will help your network generalize as it's seeing the same images but in different locations,
with different sizes, in different orientations, etc.

To randomly rotate, scale and crop, then flip your images you would define your transforms like this:

train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.5, 0.5, 0.5], 
                                                            [0.5, 0.5, 0.5])])

You'll also typically want to normalize images with transforms.Normalize. 
You pass in a list of means and list of standard deviations, then the color channels are normalized like so

input[channel] = (input[channel] - mean[channel]) / std[channel]

Subtracting mean centers the data around zero and dividing by std squishes the values 
to be between -1 and 1. Normalizing helps keep the network work weights near zero
which in turn makes backpropagation more stable. Without normalization, networks will tend to fail to learn.

You can find a list of all the available transforms here. (link : http://pytorch.org/docs/0.3.0/torchvision/transforms.html)
When you're testing however, you'll want to use images that aren't altered (except you'll need to normalize the same way). 
So, for validation/test images, you'll typically just resize and crop.

--------------------------------------------------------------------------------------------------------
"""


data_dir = 'Cat_Dog_data'

#Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                      transforms.RandomResizedCrop(224),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor()])


# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=32)
testloader = torch.utils.data.DataLoader(test_data, batch_size=32)

# change this to the trainloader or testloader 
data_iter = iter(testloader)

images, labels = next(data_iter)
fig, axes = plt.subplots(figsize=(10,4), ncols=4)
for ii in range(4):
    ax = axes[ii]
    helper.imshow(images[ii], ax=ax, normalize=False)
    
print((images.view(images.shape[0],-1)).shape) # gives output torch.Size([32, 150528])

"""
--------------------------------------------------------------------------------------------------------
In above print statement you will see that here our tensor have size ([32, 150528])
32 -batch size, 150528 -features
so we can build our model for this like below but it requires a huge amount of memory for training and testing 
more than 25GB RAM required.
--------------------------------------------------------------------------------------------------------
from torch import nn, optim
import torch.nn.functional as F

class Classifier_cat_dog(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(150528, 65535)
        self.fc2 = nn.Linear(65535, 16384)
        self.fc3 = nn.Linear(16384, 4096)
        self.fc4 = nn.Linear(4096, 1024)
        self.fc5 = nn.Linear(1024, 256)
        self.fc6 = nn.Linear(256, 128)
        self.fc7 = nn.Linear(128, 64)
        self.fc8 = nn.Linear(64, 2)
        self.drop = nn.Dropout(p=0.2)
        
    def forward(self, x):
        # make sure input tensor is flattened
        x = x.view(x.shape[0], -1)
        
        x = self.drop(F.relu(self.fc1(x)))
        x = self.drop(F.relu(self.fc2(x)))
        x = self.drop(F.relu(self.fc3(x)))
        x = self.drop(F.relu(self.fc4(x)))
        x = self.drop(F.relu(self.fc5(x)))
        x = self.drop(F.relu(self.fc6(x)))
        x = self.drop(F.relu(self.fc7(x)))
        x = F.log_softmax(self.fc8(x), dim=1)
        
        return x
        
--------------------------------------------------------------------------------------------------------
                                            Transfer Learning
--------------------------------------------------------------------------------------------------------
"""
%matplotlib inline
%config InlineBackend.figure_format = 'retina'

import matplotlib.pyplot as plt

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models

"""
--------------------------------------------------------------------------------------------------------
Most of the pretrained models require the input to be 224x224 images. 
Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately, 
the means are [0.485, 0.456, 0.406] and the standard deviations are [0.229, 0.224, 0.225].
--------------------------------------------------------------------------------------------------------

"""
data_dir = 'Cat_Dog_data'

# TODO: Define transforms for the training data and testing data
train_transforms = transforms.Compose([transforms.RandomRotation(30),
                                       transforms.RandomResizedCrop(224),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize(255),
                                      transforms.CenterCrop(224),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# Pass transforms in here, then run the next cell to see how the transforms look
train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(test_data, batch_size=64)

"""
--------------------------------------------------------------------------------------------------------

We can load in a model such as DenseNet.(link: http://pytorch.org/docs/0.3.0/torchvision/models.html#id5)
Let's print out the model architecture so we can see what's going on.
--------------------------------------------------------------------------------------------------------

"""

model = models.densenet121(pretrained=True)
print(model)

"""
--------------------------------------------------------------------------------------------------------
This model is built out of two main parts, the features and the classifier. 
The features part is a stack of convolutional layers and overall works as a feature detector that can be fed into a classifier.
The classifier part is a single fully-connected layer (classifier): Linear(in_features=1024, out_features=1000). 
This layer was trained on the ImageNet dataset, so it won't work for our specific problem. 
That means we need to replace the classifier, but the features will work perfectly on their own. In general, 
I think about pre-trained networks as amazingly good feature detectors that can be used as the input for 
simple feed-forward classifiers.
--------------------------------------------------------------------------------------------------------
"""

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False

from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([
                          ('fc1', nn.Linear(1024, 500)),
                          ('relu', nn.ReLU()),
                          ('fc2', nn.Linear(500, 2)),
                          ('output', nn.LogSoftmax(dim=1))
                          ]))
    
model.classifier = classifier

"""
--------------------------------------------------------------------------------------------------------
With our model built, we need to train the classifier. However, now we're using a really deep neural network. 
If you try to train this on a CPU like normal, it will take a long, long time. 
Instead, we're going to use the GPU to do the calculations. 
The linear algebra computations are done in parallel on the GPU leading to 100x increased training speeds. 
It's also possible to train on multiple GPUs, further decreasing training time.
PyTorch, along with pretty much every other deep learning framework, 
uses CUDA to efficiently compute the forward and backwards passes on the GPU. 
In PyTorch, you move your model parameters and other tensors to the GPU memory using model.to('cuda'). 
You can move them back from the GPU with model.to('cpu') which you'll commonly do when you need to operate
on the network output outside of PyTorch.
--------------------------------------------------------------------------------------------------------
"""
# Use GPU if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = models.densenet121(pretrained=True)

# Freeze parameters so we don't backprop through them
for param in model.parameters():
    param.requires_grad = False
    
model.classifier = nn.Sequential(nn.Linear(1024, 256),
                                 nn.ReLU(),
                                 nn.Dropout(0.2),
                                 nn.Linear(256, 2),
                                 nn.LogSoftmax(dim=1))

criterion = nn.NLLLoss()

# Only train the classifier parameters, feature parameters are frozen
optimizer = optim.Adam(model.classifier.parameters(), lr=0.003)

model.to(device);

"""
--------------------------------------------------------------------------------------------------------
"""

epochs = 1
steps = 0
running_loss = 0
print_every = 5
for epoch in range(epochs):
    for inputs, labels in trainloader:
        steps += 1
        # Move input and label tensors to the default device
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
                for inputs, labels in testloader:
                    inputs, labels = inputs.to(device), labels.to(device)
                    logps = model.forward(inputs)
                    batch_loss = criterion(logps, labels)
                    
                    test_loss += batch_loss.item()
                    
                    # Calculate accuracy
                    ps = torch.exp(logps)
                    top_p, top_class = ps.topk(1, dim=1)
                    equals = top_class == labels.view(*top_class.shape)
                    accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Epoch {epoch+1}/{epochs}.. "
                  f"Train loss: {running_loss/print_every:.3f}.. "
                  f"Test loss: {test_loss/len(testloader):.3f}.. "
                  f"Test accuracy: {accuracy/len(testloader):.3f}")
            running_loss = 0
            model.train()
