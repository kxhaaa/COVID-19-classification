# -*- coding: utf-8 -*-
"""
Created on Thu Jul 23 16:10:59 2020

@author: 孔湘涵
"""
from __future__ import print_function, division
import torch

import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.nn as nn
import time
import numpy as np
import copy
from sklearn.metrics import roc_curve, auc, roc_auc_score, f1_score
def train( model, optimizer, scheduler, loss_fn, train_loader, val_loader, path, epochs, device="cpu"):
    full_name=path+'/record.txt'
    file = open(full_name, 'w+')
    since = time.time()
    Loss_list1 = []
    Accuracy_list1 = []
    Loss_list2 = []
    Accuracy_list2 = []
    best_model_wts = copy.deepcopy(model.state_dict())
    best_f1 = 0.0
    best_epoch=0
    for epoch in range(epochs):
        training_loss=0.0
        valid_loss=0.0
# =============================================================================
#         training
# =============================================================================
        model.train()
        i=0
        total_trainnum = 0.0
        num_correct=0
        tra_true=[]
        tra_score=[]
        tra_pred=[]

        for batch in train_loader:
            i+=1
            optimizer.zero_grad()
            inputs, target = batch
            inputs = inputs.to(device)
#            if i%10==0:
#                print('Epoch:',epoch,'batch:',i)
            target = target.to(device)
#            print(inputs.size())
            output = model(inputs)
#            print(torch.max(output, dim=1))
#            print(target)
#            print(outputs)
            
            if str(loss_fn)=='MSELoss()':  
                outputs, predicted = torch.max(output, dim=1)          
            if str(loss_fn)=='CrossEntropyLoss()':
                target = target.long()
                outputs = output.float()
                _, predicted = torch.max(output, dim=1)
            if str(loss_fn)=='BCELoss()':
                outputs[outputs < 0.0] = 0.0
                outputs[outputs > 1.0] = 1.0
                _, predicted = torch.max(output, dim=1)

            loss = loss_fn(outputs, target)      #输入图像和标签，通过infer计算得到预测值，计算损失函数
            outputs,_ = torch.max(output, dim=1)

            loss.backward()     #反向传播，计算当前梯度
            optimizer.step()    #更新网络参数，使用计算梯度来调整权重
            scheduler.step()    #调整lr
            training_loss +=loss.data.item()
#            print(target)
#
#            print(torch.max(output,1))
#            correct = (target == torch.max(output, 1)[1].float()).sum()
#            num_correct += correct.data.item()
            total_trainnum += target.size(0)
            num_correct += (predicted == target).sum().item()
            for j in range(target.size(0)):

                tra_true.append(target[j].item())
                tra_score.append(outputs[j].item())
                tra_pred.append(predicted[j].item())
        tra_score=np.array(tra_score)
        tra_true=np.array(tra_true)
        tra_pred=np.array(tra_pred)

        training_loss/=i  #求一个batch的loss，除的是mini batch的个数
        train_acc=num_correct/total_trainnum
        roc_auc2 = roc_auc_score(tra_true,tra_score)
#        tra_f1=f1_score(tra_true, tra_pred, pos_label=1)

# =============================================================================
#         evaluation
# =============================================================================
        model.eval()    #约等于model.train()，但不启用normalization和dropout       
        num_correct2 = 0
        total_valnum2 = 0
        i=0
#        TP,FN,FP=0,0,0
        y_true=[]
        y_score=[]
        y_pred=[]
        for batch in val_loader:
            i+=1
            inputs, target = batch
            inputs = inputs.to(device)
            output = model(inputs).float()
            target = target.to(device)
            
            if str(loss_fn)=='MSELoss()':  
                outputs, predicted = torch.max(output, dim=1)  
                loss_val = loss_fn(outputs, target)      #输入图像和标签，通过infer计算得到预测值，计算损失函数
            if str(loss_fn)=='CrossEntropyLoss()':
                target = target.long()
                outputs = output.float()
                _, predicted = torch.max(output, dim=1)
                loss_val = loss_fn(outputs, target)      #输入图像和标签，通过infer计算得到预测值，计算损失函数
            if str(loss_fn)=='BCELoss()':
                outputs[outputs < 0.0] = 0.0
                outputs[outputs > 1.0] = 1.0
                _, predicted = torch.max(output, dim=1)
                loss_val = loss_fn(outputs, target)      #输入图像和标签，通过infer计算得到预测值，计算损失函数

            outputs,_ = torch.max(output, dim=1)
            valid_loss += loss_val.data.item()
            correct2 = torch.eq(predicted,target).sum() #追踪最高的准确率
            num_correct2 += correct2.data.item() #计数output和target相等的个数
            total_valnum2 += target.size(0)

            for j in range(target.size(0)):
                #COVID-19 is labeled 0
#                if (predicted[j].item() == 0) & (target[j].item() == 0):
#                    TP += 1
#                if (predicted[j].item() == 0) & (target[j].item() == 1):
#                    FP += 1
#                if (predicted[j].item() == 1) & (target[j].item() == 0):
#                    FN += 1
                
                y_true.append(target[j].item())
                y_score.append(outputs[j].item())
                y_pred.append(predicted[j].item())
#        precision = TP/(TP+FP)
#        recall = TP/(TP+FN)
#        F1_socre = 2*precision*recall/(precision+recall)
        
        y_score=np.array(y_score)
        y_true=np.array(y_true)
        y_pred=np.array(y_pred)
#        y_score[y_score < 0.0] = 0.0
#        y_score[y_score > 1.0] = 1.0
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
        roc_auc = auc(fpr, tpr) 
        f1=f1_score(y_true, y_pred, pos_label=1)
        
        valid_loss /=i
        val_acc = num_correct2/total_valnum2
        
        if (epoch > 100) & (f1 >= best_f1):
            best_f1 = f1
            best_epoch = epoch
            loss_best=valid_loss
            auc_best=roc_auc
            acc_best=val_acc
            fpr_best,tpr_best = fpr,tpr
            best_model_wts = copy.deepcopy(model.state_dict())

        print('Epoch {},Training Loss:{:.2f}%,Validation Loss:{:.2f}%,\
accuracy:{:.2f}%'.format(epoch, 100*training_loss, 100*valid_loss, 100*val_acc))
        print('F1 socre:{:.4f}, AUC:{:.4f}, Trainset AUC:{:.4f}'
              .format(f1,roc_auc,roc_auc2),'\n')
        file.write('Epoch {},Training Loss:{:.2f}%,Validation Loss:{:.2f}%,accuracy:{:.2f}%'
                   .format(epoch, 100*training_loss, 100*valid_loss, 100*val_acc)+'\n')
        file.write('    F1 socre:{:.4f}, AUC:{:.4f}, Trainset AUC:{:.4f}'
              .format(f1,roc_auc,roc_auc2)+'\n'+'\n')
        Loss_list1.append(100*training_loss)
        Accuracy_list1.append(100 * train_acc)
        Loss_list2.append(100*valid_loss)
        Accuracy_list2.append(100 * val_acc)
    
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('The best f1_socre is {:.4f} in epoch {}. Test loss is {:.2f}%, acc is {:.2f}%, auc is {:.4f}'
          .format(best_f1,best_epoch,100*loss_best,100*acc_best,auc_best))
    file.write('\n'+'Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60)+'\n')
    file.write('The best f1_socre is {:.4f} in epoch {}. Test loss is {:.2f}%, acc is {:.2f}%, auc is {:.4f}'
          .format(best_f1,best_epoch,100*loss_best,100*acc_best,auc_best)+'\n')
    file.close()
#    if file.closed:
#        print('file is closed.')
#    else:
#        print('file is not closed.')
    model.load_state_dict(best_model_wts)
# =============================================================================
#   drawing loss & acc curve
# =============================================================================
    x1 = range(0, epochs)
    x2 = range(0, epochs)
    y1 = Accuracy_list1
    y2 = Loss_list1
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x1, y1, 'o-')
    plt.title('Train accuracy vs. epoches')
    plt.ylabel('Train accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x2, y2, '.-')
    plt.xlabel('Train loss vs. epoches')
    plt.ylabel('Train loss')
    plt.savefig((path+'/'+'Train accuracy_loss.png'))
    plt.show()

    x3 = range(0, epochs)
    x4 = range(0, epochs)
    y3 = Accuracy_list2
    y4 = Loss_list2
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.plot(x3, y3, 'o-')
    plt.title('Test accuracy vs. epoches')
    plt.ylabel('Test accuracy')
    plt.subplot(2, 1, 2)
    plt.plot(x4, y4, '.-')
    plt.xlabel('Test loss vs. epoches')
    plt.ylabel('Test loss')
    plt.savefig((path+'/'+'Test accuracy_loss.png'))
    plt.show()

    plt.figure()
    plt.plot(fpr_best,tpr_best,'o-')
    plt.title('Receiver operating characteristic curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.savefig((path+'/'+'ROC curve.png'))
    plt.show()

    return model


