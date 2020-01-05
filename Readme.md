# cat - dog classification or identification using Transfer Learning in PyTorch

# Transfer Learning

<h2>ImageNet</h2> is a massive dataset with over 1 million labeled images in 1000 categories. It's used to train deep neural networks using an architecture called convolutional layers. I'm not going to get into the details of convolutional networks here, but if you want to learn more about them, please watch this.


With torchvision.models you can download these pre-trained networks and use them in your applications. 

# Loading Image Data

We'll be using a dataset of cat and dog photos available from Kaggle.
link : https://www.kaggle.com/c/dogs-vs-cats


# Transforms
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
    
    
# Data Loaders

    With the ImageFolder loaded, you have to pass it to a DataLoader.
    
    The DataLoader takes a dataset (such as you would get from ImageFolder) and returns batches of images and the corresponding labels. You can set various parameters like the batch size and if the data is shuffled after each epoch.
    
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=32, shuffle=True)
    Here dataloader is a generator. To get data out of it, you need to loop through it or convert it to an iterator and call next().

# Data Augmentation

    A common strategy for training neural networks is to introduce randomness in the input data itself. 
    For example, you can randomly rotate, mirror, scale, and/or crop your images during training.
    This will help your network generalize as it's seeing the same images but in different locations,
    with different sizes, in different orientations, etc.
    
# Building our own model
    32 -batch size, 150528 -features
    so we can build our model for this like below but it requires a huge amount of memory for training and testing 
    more than 25GB RAM required. 
    as we don't have that huge specification of hardware we will use pretrained model as discusse above.
    
# Transfer Learning

    Most of the pretrained models require the input to be 224x224 images. 
    Also, we'll need to match the normalization used when the models were trained. Each color channel was normalized separately, 
    the means are [0.485, 0.456, 0.406] and the standard deviations are [0.229, 0.224, 0.225].
    DenseNet.(link: http://pytorch.org/docs/0.3.0/torchvision/models.html#id5)
    
          
