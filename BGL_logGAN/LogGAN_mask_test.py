import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Model.LogGAN_mask import G_Model,D_Model
import csv
import os


h_window=1
hidden_size=128
num_layers=2
num_keys=394
emb_dimension=200
topK=10
train_flag=False
threshold=[0.95,0.9,0.85,0.8,0.75,0.7,0.65,0.6,0.55,0.5,0.45,0.40,0.35,0.3,0.25,0.2,0.15,0.1]
torch.cuda.set_device(1)
session_data = []
f = open('Dataset/test_data_session_tmp.csv', 'r')
reader = csv.reader(f)
for line in reader:
    data = []
    if len(line) < h_window + 1:
        continue
    for v in line:
        v = int(v) - 1
        data.append(v)
    session_data.append(data)
session_label = []
f = open('Dataset/test_data_label_tmp.csv', 'r')
reader = csv.reader(f)
for line in reader:
    if len(line) < h_window + 1:
        continue
    data = []
    for v in line:
        v = int(v)
        data.append(v)
    session_label.append(data)

test_tuple = {}
for i in range(len(session_data)):
    session = session_data[i]
    for j in range(len(session) - h_window - 1):
        input = session[j:j + h_window]
        label = session_label[i][j + h_window + 1]
        target = session[j + h_window + 1]
        key = tuple(input + [target])
        test_tuple.setdefault(key, {'frequence': 0, 'normal': 0, 'anormal': 0})
        test_tuple[key]['frequence'] += 1
        if label:
            test_tuple[key]['normal'] += 1
        else:
            test_tuple[key]['anormal'] += 1

with torch.no_grad():
    model=G_Model(emb_dimension,hidden_size,num_layers,num_keys,emb_dimension).cuda()
    model.load_state_dict(torch.load('Checkpoint/G_model_mask_tmp_window1.pkl'))
    print('模型初始化完成')
    TP=[]
    FP=[]
    FN=[]
    TN=[]
    precision=[]
    recall=[]
    for i in range(len(threshold)):
        TP.append(0)
        FP.append(0)
        FN.append(0)
        TN.append(0)
        precision.append(0)
        recall.append(0)
    for key in test_tuple.keys():
        data=list(key)
        input=data[0:h_window]
        target=data[-1]
        input_tensor = Variable(torch.LongTensor([input]))
        input_tensor.cuda()
        y = model(input_tensor, train_flag)
        y = y.view(-1)
        y = y.cpu()
        r = float(y[target])
        for k in range(len(threshold)):
            if r >= threshold[k]:
                FN[k] += test_tuple[key]['anormal']
                TN[k] += test_tuple[key]['normal']
            else:
                TP[k] += test_tuple[key]['anormal']
                FP[k] += test_tuple[key]['normal']

    F1=[]
    TNR=[]
    for k in range(len(TP)):
        precision[k]=float(TP[k])/max((TP[k]+FP[k]),1)
        recall[k] = float(TP[k]) / max((TP[k] + FN[k]),1)
        F1.append(2 * precision[k] * recall[k] / max((precision[k] + recall[k]),0.001))
        TNR.append(float(TN[k])/max((TN[k]+FN[k]),1))
print(precision)
print(recall)
print(F1)
print(TNR)
f=open('Result/result_window1.csv','w',newline='')
writer=csv.writer(f)
writer.writerow(TP)
writer.writerow(FP)
writer.writerow(TN)
writer.writerow(FN)
writer.writerow(precision)
writer.writerow(recall)
writer.writerow(F1)
writer.writerow(TNR)
