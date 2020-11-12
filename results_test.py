# -*- coding: utf-8 -*-
"""
Created on Tue Nov  3 11:10:37 2020

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
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score

from torchsummary import summary
import torch.nn.functional as F

path='./10.29_14.45/'
model = torch.load(path+'model.pkl')
model.eval()

root='./Images-processed/'
dataset=torchvision.datasets.ImageFolder(root,transform=transforms.Compose([
        transforms.Resize([256,256]),
#        transforms.RandomHorizontalFlip(p = 0.5 ),
#        transforms.ColorJitter(brightness=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
        ]))
'''
test set
'''
indices=np.load(path+'valset_indices.npy')
indices=indices.tolist()
dataset=torch.utils.data.Subset(dataset, indices)
dataloader= DataLoader(dataset,batch_size=15)#,drop_last=True)

print('length of testset is ',len(dataset))
images,labels=next(iter(dataloader))
img=torchvision.utils.make_grid(images)
img=img.numpy().transpose(1,2,0)
std=[0.5,0.5,0.5]
mean=[0.5,0.5,0.5]
img=img*std+mean
plt.imshow(img)
plt.show()

num_correct = 0
total_num = 0
y_true=[]
y_score=[]
y_pred=[]

for batch in dataloader:
    inputs, target = batch
    inputs = inputs.to('cuda')
    output = model(inputs).float()
    target = target.to('cuda')

    target = target.long()
    outputs, predicted = torch.max(output, dim=1)

    correct = torch.eq(predicted,target).sum() #追踪最高的准确率
    num_correct += correct.data.item() #计数output和target相等的个数
    total_num += target.size(0)

    for j in range(target.size(0)):                
        y_true.append(target[j].item())
        y_score.append(outputs[j].item())
        y_pred.append(predicted[j].item())

y_score=np.array(y_score)
y_true=np.array(y_true)
y_pred=np.array(y_pred)
fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
roc_auc = auc(fpr, tpr) 
f1=f1_score(y_true, y_pred, pos_label=1)
acc = num_correct/total_num

print('F1 socre:{:.4f}, AUC:{:.4f}, ACC:{:.4f}'.format(f1,roc_auc,acc))

plt.figure()
plt.plot(fpr,tpr,'o-')
plt.title('Receiver operating characteristic curve')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.show()


