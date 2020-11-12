# -*- coding: utf-8 -*-
"""
Created on Mon Oct  5 16:52:55 2020

@author: 孔湘涵
"""

from __future__ import print_function, division
import torchvision
import torch
import os
import time
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn as nn
import torch.optim as optim
from trainer import train
from modelvisualize import visualize_model

from torchsummary import summary
import torch.nn.functional as F

# =============================================================================
# #define the network
# =============================================================================
'''
resnet34
'''
model = torchvision.models.resnet34(True)

#freeze the model
#for param in model.parameters():
#    param.requires_grad = False
#model.fc = nn.Sequential(nn.Linear(model.fc.in_features,512),
#                                  nn.ReLU(),
#                                  nn.Dropout(p=0.7),
#                                  nn.Linear(512,2)) 
#model.fc = nn.Linear(model.fc.in_features, 2)
model.fc = nn.Sequential(
        nn.BatchNorm1d(512),
        nn.Linear(512, 256),
        nn.ReLU(),
        nn.BatchNorm1d(256),
        nn.Linear(256, 128),
        nn.ReLU(),
        nn.BatchNorm1d(128),
        nn.Linear(128, 64),
        nn.ReLU(),
        nn.BatchNorm1d(64),
        nn.Linear(64, 2)
        )

'''
resnet50
'''
#model = torchvision.models.resnet50(True)
#model.fc = nn.Sequential(
#        nn.BatchNorm1d(2048),
#        nn.Linear(2048,1024),
#        nn.ReLU(),
#        nn.BatchNorm1d(1024),
#        nn.Linear(1024, 512),
#        nn.ReLU(),
#        nn.BatchNorm1d(512),
#        nn.Linear(512, 256),
#        nn.ReLU(),
#        nn.BatchNorm1d(256),
#        nn.Linear(256, 2)   
##        nn.ReLU(),
##        nn.BatchNorm1d(128),
##        nn.Linear(128, 64),#15.02
##        nn.ReLU(),
##        nn.BatchNorm1d(64),
##        nn.Linear(64, 32), 
##        nn.ReLU(),
##        nn.BatchNorm1d(32),
##        nn.Linear(32, 2)       
#        )
'''
resnet18
'''     
#model = torchvision.models.resnet18(True)
#model.fc = nn.Sequential(
#        nn.BatchNorm1d(512),
#        nn.Linear(512,256),
#        nn.ReLU(),
#        nn.BatchNorm1d(256),
#        nn.Linear(256, 128),     
#        nn.ReLU(),
#        nn.BatchNorm1d(128),
#        nn.Linear(128, 64),
#        nn.ReLU(),
#        nn.BatchNorm1d(64),
#        nn.Linear(64, 2)        
#        )

model=model.to('cuda')
print('Model is useful',model.training)
#inputs=torch.rand(1,50, 60)
#print(inputs.shape)
#
#output=model(inputs)
#inputs = (3, 224, 224)
#
#summary(model, inputs)

# =============================================================================
# load the dataset
# =============================================================================
#root1='./Images-processed/'
#root2='./validation/'
#root1='E:/pytorch_learn/train/'
#root2='E:/pytorch_learn/val/'

#traindataset=torchvision.datasets.ImageFolder(root1,transform=transforms.Compose([
#     transforms.Resize([256,256]),
#     transforms.RandomHorizontalFlip(p = 0.5 ),
##     transforms.RandomRotation(degrees=90),
##     transforms.Grayscale(num_output_channels=3),
##     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
# ]))
#train_data_loader=DataLoader(traindataset,batch_size=32,drop_last=True,shuffle=True)

root='./Images-processed/'
#root='E:/pytorch_learn/dogs-vs-cats/'
dataset=torchvision.datasets.ImageFolder(root,transform=transforms.Compose([
     transforms.Resize([256,256]),
     transforms.RandomHorizontalFlip(p = 0.5 ),
#     transforms.RandomRotation(degrees=90),
#     transforms.Grayscale(num_output_channels=3),
     transforms.ColorJitter(brightness=0.2),
     transforms.ToTensor(),
     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
#     transforms.Normalize(mean = (0.5, 0.5, 0.5), std = (0.5, 0.5, 0.5))
 ]))
dataset1=torchvision.datasets.ImageFolder(root,transform=transforms.Compose([
        transforms.Resize([256,256]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]))


n_train = len(dataset)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size
traindataset,valdataset = torch.utils.data.random_split(dataset, [train_size, test_size])
valdataset=torch.utils.data.Subset(dataset1,valdataset.indices)
'''
load dataset from previous file
'''
#indices=np.load('./10.29_14.45/trainset_indices.npy')
#indices=indices.tolist()
#traindataset=torch.utils.data.Subset(dataset, indices)
#indices=np.load('./10.29_14.45/valset_indices.npy')
#indices=indices.tolist()
#valdataset=torch.utils.data.Subset(dataset1,indices)

train_data_loader=DataLoader(traindataset,batch_size=32,drop_last=True,shuffle=True)
val_data_loader=DataLoader(valdataset,batch_size=32,drop_last=True,shuffle=True)

images,labels=next(iter(train_data_loader))
img=torchvision.utils.make_grid(images)
img=img.numpy().transpose(1,2,0)
std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
img=img*std+mean
print([labels[i] for i in range(16)])
plt.imshow(img)
plt.show()

class_names = dataset.classes
print('There is {} class:'.format(len(class_names)),class_names,',{} traindata & {} valdata'
      .format(len(traindataset),len(valdataset)))
#valdataset=torchvision.datasets.ImageFolder(root2,transform=transforms.Compose([
#     transforms.Resize([256,256]),
#     transforms.RandomHorizontalFlip(p = 0.5 ),
##     transforms.RandomRotation(degrees=90),
##     transforms.Grayscale(num_output_channels=3),
##     transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
# ]))

# =============================================================================
# super parameter & train
# =============================================================================
#finetuning
optimizer = optim.Adam(model.parameters(),lr=0.0003,weight_decay=1e-3)
#fixed feature extractor
#optimizer = optim.Adam(model.fc.parameters(),lr=0.003,weight_decay=1e-3)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.99)
loss_fn=nn.CrossEntropyLoss()
#loss_fn=nn.MSELoss()
#loss_fn=nn.BCELoss()

#path
time1=time.localtime()
path=str(time1.tm_mon)+'.'+str(time1.tm_mday)+'_'+str(time1.tm_hour)+'.'+str(time1.tm_min)
#path='test'
os.makedirs(path)

x=np.array(traindataset.indices)
np.save((path+'/trainset_indices.npy'),x)
y=np.array(valdataset.indices)
np.save((path+'./valset_indices.npy'),y)
#a=np.load(‘a.npy’)
# a=a.tolist()
model = train( model, optimizer, scheduler, loss_fn, train_data_loader, val_data_loader, path, epochs=500, device="cuda")

visualize_model(model,val_data_loader,class_names)
# =============================================================================
# save model
# =============================================================================
#save parameters
torch.save(model.state_dict(), (path+'/'+'params.pkl'))
#model.load_state_dict(torch.load('params.pkl'))
#model.eval()

#save the whole model
torch.save(model, (path+'/'+'model.pkl'))
#model = torch.load('model.pkl')
#model.eval()




