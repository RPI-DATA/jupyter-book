---
interact_link: content/C:\Users\sjgar\Documents\GitHub\jupyter-book\content\notebooks/pytorch.ipynb
kernel_name: python3
has_widgets: false
title: 'PyTorch'
prev_page:
  url: /notebooks/how-to
  title: 'Getting Started'
next_page:
  url: /notebooks/linearregression
  title: 'Linear Regression'
comment: "***PROGRAMMATICALLY GENERATED, DO NOT EDIT. SEE ORIGINAL FILES IN /content***"
---


[![AnalyticsDojo](https://github.com/rpi-techfundamentals/fall2018-materials/blob/master/fig/final-logo.png?raw=1)](http://rpi.analyticsdojo.com)

<center><h1>PyTorch with the MNIST Dataset</h1></center>

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/04_pytorch_mnist.ipynb)



**PyTorch** is a open-source library that takes the machine learning tools of Torch and adapts them for use in Python. The following code is adopted from the [PyTorch examples repository](https://github.com/pytorch/examples/). It is licensed under [BSD 3-Clause "New" or "Revised" License](https://github.com/pytorch/examples/blob/master/LICENSE).

This notebook uses the following pedagogical patterns:
* [**4.2** Shift-enter for the win](https://jupyter4edu.github.io/jupyter-edu-book/catalogue.html#shift-enter-for-the-win)
* [**4.15** The world is your dataset](https://jupyter4edu.github.io/jupyter-edu-book/catalogue.html#the-world-is-your-dataset)



## Learning Objectives
---
1. Learn how to utilize PyTorch
2. Employ Pytorch in the creation of an image-recognition algorithm



## Problem Definition
---
To illustrate the computational power of PyTorch, we will take a crack at processing the MNIST database. **MNIST** is a database of 70,000 images of handwritten numbers used to evaluate image processing techniques. From [Kaggle](https://www.kaggle.com/c/digit-recognizer): 
> MNIST ("Modified National Institute of Standards and Technology") is the de facto "hello world" dataset of computer vision. Since its release in 1999, this classic dataset of handwritten images has served as the basis for benchmarking classification algorithms. As new machine learning techniques emerge, MNIST remains a reliable resource for researchers and learners alike.

<a title="By Josef Steppan [CC BY-SA 4.0 (https://creativecommons.org/licenses/by-sa/4.0)], from Wikimedia Commons" href="https://commons.wikimedia.org/wiki/File:MnistExamples.png"><img width="512" alt="MnistExamples" src="https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png"/></a>



Check out this table from [Wikipedia](https://en.wikipedia.org/wiki/MNIST_database) to see what kind of machine learning methods generate good error rates.



## PyTorch vs. Tensorflow
---
PyTorch serves as a great tool for learning data science because of its flexibility when compared to other libraries. Below are a few points of comparison between PyTorch and another popular dataflow tool, Tensorflow:
- PyTorch enables dynamic computational graphs, while Tensorflow's computation is static. This means that at runtime PyTorch defines the graph's structure, which can be changed depending on parameters like the input data. Conversely, Tensorflow needs to have the structure defined _before_ running. 
- Tensorflow enables easier deployment and requires less memory because it only has to worry about computations at the end.



## Setting up PyTorch
---
Start by installing PyTorch with the following command:



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
!pip install torch torchvision

```
</div>

</div>



We will then import all of the libraries needed for our algorithm.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

```
</div>

</div>



Next we can define the arguments for our functions and then load in our data.



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
args={}
kwargs={}
args['batch_size']=1000
args['test_batch_size']=1000
args['epochs']=10     #The number of Epochs is the number of times you go through the full dataset. 
args['lr']=0.01       #Learning rate is how fast it will decend. 
args['momentum']=0.5  #SGD momentum (default: 0.5) Momentum is a moving average of our gradients (helps to keep direction).

args['seed']=1        #random seed
args['log_interval']=10
args['cuda']=False

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
#load the data
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['batch_size'], shuffle=True, **kwargs)
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))
                   ])),
    batch_size=args['test_batch_size'], shuffle=True, **kwargs)


```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Downloading http://yann.lecun.com/exdb/mnist/train-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/train-labels-idx1-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-images-idx3-ubyte.gz
Downloading http://yann.lecun.com/exdb/mnist/t10k-labels-idx1-ubyte.gz
Processing...
Done!
```
</div>
</div>
</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
class Net(nn.Module):
    #This defines the structure of the NN.
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()  #Dropout
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        #Convolutional Layer/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv1(x), 2)) 
        #Convolutional Layer/Dropout/Pooling Layer/Activation
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        #Fully Connected Layer/Activation
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        #Fully Connected Layer/Activation
        x = self.fc2(x)
        #Softmax gets probabilities. 
        return F.log_softmax(x, dim=1)

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        #Variables in Pytorch are differenciable. 
        data, target = Variable(data), Variable(target)
        #This will zero out the gradients for this batch. 
        optimizer.zero_grad()
        output = model(data)
        # Calculate the negative log likelihood loss - it's useful to train a classification problem with C classes.
        loss = F.nll_loss(output, target)
        #dloss/dx for every Variable 
        loss.backward()
        #to do a one-step update on our parameter.
        optimizer.step()
        #Print out the loss periodically. 
        if batch_idx % args['log_interval'] == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.data[0]))

def test():
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        if args['cuda']:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        output = model(data)
        test_loss += F.nll_loss(output, target, size_average=False).data[0] # sum up batch loss
        pred = output.data.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.data.view_as(pred)).long().cpu().sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

```
</div>

</div>



<div markdown="1" class="cell code_cell">
<div class="input_area" markdown="1">
```python
model = Net()
if args['cuda']:
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=args['lr'], momentum=args['momentum'])

for epoch in range(1, args['epochs'] + 1):
    train(epoch)
    test()

```
</div>

<div class="output_wrapper" markdown="1">
<div class="output_subarea" markdown="1">
{:.output_stream}
```
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.338192
Train Epoch: 1 [10000/60000 (17%)]	Loss: 2.305725
Train Epoch: 1 [20000/60000 (33%)]	Loss: 2.289212
Train Epoch: 1 [30000/60000 (50%)]	Loss: 2.283156
Train Epoch: 1 [40000/60000 (67%)]	Loss: 2.270567
Train Epoch: 1 [50000/60000 (83%)]	Loss: 2.261630

Test set: Average loss: 2.2199, Accuracy: 3655/10000 (37%)

Train Epoch: 2 [0/60000 (0%)]	Loss: 2.240778
Train Epoch: 2 [10000/60000 (17%)]	Loss: 2.209014
Train Epoch: 2 [20000/60000 (33%)]	Loss: 2.170792
Train Epoch: 2 [30000/60000 (50%)]	Loss: 2.140490
Train Epoch: 2 [40000/60000 (67%)]	Loss: 2.080513
Train Epoch: 2 [50000/60000 (83%)]	Loss: 1.990342

Test set: Average loss: 1.7368, Accuracy: 7205/10000 (72%)

Train Epoch: 3 [0/60000 (0%)]	Loss: 1.924992
Train Epoch: 3 [10000/60000 (17%)]	Loss: 1.759480
Train Epoch: 3 [20000/60000 (33%)]	Loss: 1.636611
Train Epoch: 3 [30000/60000 (50%)]	Loss: 1.517218
Train Epoch: 3 [40000/60000 (67%)]	Loss: 1.348585
Train Epoch: 3 [50000/60000 (83%)]	Loss: 1.313530

Test set: Average loss: 0.8124, Accuracy: 8438/10000 (84%)

Train Epoch: 4 [0/60000 (0%)]	Loss: 1.169621
Train Epoch: 4 [10000/60000 (17%)]	Loss: 1.145530
Train Epoch: 4 [20000/60000 (33%)]	Loss: 1.056403
Train Epoch: 4 [30000/60000 (50%)]	Loss: 0.992876
Train Epoch: 4 [40000/60000 (67%)]	Loss: 0.980686
Train Epoch: 4 [50000/60000 (83%)]	Loss: 0.950357

Test set: Average loss: 0.5138, Accuracy: 8800/10000 (88%)

Train Epoch: 5 [0/60000 (0%)]	Loss: 0.930668
Train Epoch: 5 [10000/60000 (17%)]	Loss: 0.879105
Train Epoch: 5 [20000/60000 (33%)]	Loss: 0.874244
Train Epoch: 5 [30000/60000 (50%)]	Loss: 0.787681
Train Epoch: 5 [40000/60000 (67%)]	Loss: 0.814346
Train Epoch: 5 [50000/60000 (83%)]	Loss: 0.779896

Test set: Average loss: 0.4082, Accuracy: 8966/10000 (90%)

Train Epoch: 6 [0/60000 (0%)]	Loss: 0.744148
Train Epoch: 6 [10000/60000 (17%)]	Loss: 0.730266
Train Epoch: 6 [20000/60000 (33%)]	Loss: 0.730913
Train Epoch: 6 [30000/60000 (50%)]	Loss: 0.697980
Train Epoch: 6 [40000/60000 (67%)]	Loss: 0.736012
Train Epoch: 6 [50000/60000 (83%)]	Loss: 0.711165

Test set: Average loss: 0.3525, Accuracy: 9069/10000 (91%)

Train Epoch: 7 [0/60000 (0%)]	Loss: 0.722657
Train Epoch: 7 [10000/60000 (17%)]	Loss: 0.652839
Train Epoch: 7 [20000/60000 (33%)]	Loss: 0.716362
Train Epoch: 7 [30000/60000 (50%)]	Loss: 0.678424
Train Epoch: 7 [40000/60000 (67%)]	Loss: 0.665473
Train Epoch: 7 [50000/60000 (83%)]	Loss: 0.614177

Test set: Average loss: 0.3153, Accuracy: 9121/10000 (91%)

Train Epoch: 8 [0/60000 (0%)]	Loss: 0.621331
Train Epoch: 8 [10000/60000 (17%)]	Loss: 0.550397
Train Epoch: 8 [20000/60000 (33%)]	Loss: 0.623889
Train Epoch: 8 [30000/60000 (50%)]	Loss: 0.609498
Train Epoch: 8 [40000/60000 (67%)]	Loss: 0.632714
Train Epoch: 8 [50000/60000 (83%)]	Loss: 0.567455

Test set: Average loss: 0.2897, Accuracy: 9188/10000 (92%)

Train Epoch: 9 [0/60000 (0%)]	Loss: 0.637325
Train Epoch: 9 [10000/60000 (17%)]	Loss: 0.607037
Train Epoch: 9 [20000/60000 (33%)]	Loss: 0.607436
Train Epoch: 9 [30000/60000 (50%)]	Loss: 0.605397
Train Epoch: 9 [40000/60000 (67%)]	Loss: 0.540220
Train Epoch: 9 [50000/60000 (83%)]	Loss: 0.567621

Test set: Average loss: 0.2713, Accuracy: 9224/10000 (92%)

Train Epoch: 10 [0/60000 (0%)]	Loss: 0.538887
Train Epoch: 10 [10000/60000 (17%)]	Loss: 0.529944
Train Epoch: 10 [20000/60000 (33%)]	Loss: 0.570023
Train Epoch: 10 [30000/60000 (50%)]	Loss: 0.558310
Train Epoch: 10 [40000/60000 (67%)]	Loss: 0.513574
Train Epoch: 10 [50000/60000 (83%)]	Loss: 0.528905

Test set: Average loss: 0.2524, Accuracy: 9284/10000 (93%)

```
</div>
</div>
</div>



[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/RPI-DATA/tutorials-intro/blob/master/website/04_pytorch_mnist.ipynb)

