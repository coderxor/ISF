import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import torch.optim as optim
from Model.LogGAN_mask import G_Model,D_Model
import csv
import os

def Test():
    f_data = open('Dataset/test_data_session_tmp.csv', 'r')
    reader_data = csv.reader(f_data)

    f_label = open('Dataset/test_data_label_tmp.csv', 'r')
    reader_label = csv.reader(f_label)
    index=0
    for Sessen,lables in zip(reader_data,reader_label):
        print(index)
        index+=1
        for i in range(len(Sessen)):
            if i<5: continue
            tuple2=  list(map(lambda x: int(x),Sessen[i-2:i]))
            tuple3 = list(map(lambda x:int(x),Sessen[i - 3:i]))
            tuple4 = list(map(lambda x:int(x),Sessen[i - 4:i]))
            tuple5 = list(map(lambda x: int(x),Sessen[i - 5:i]))
            y2 = model_window2(Variable(torch.LongTensor([tuple2])).cuda(), train_flag)
            r2 = float(y2.view(-1).cpu()[int(Sessen[i])])

            y3 = model_window3(Variable(torch.LongTensor([tuple3])).cuda(), train_flag)
            r3 = float(y3.view(-1).cpu()[int(Sessen[i])])

            y4 = model_window4(Variable(torch.LongTensor([tuple4])).cuda(), train_flag)
            r4 = float(y4.view(-1).cpu()[int(Sessen[i])])

            y5 = model_window5(Variable(torch.LongTensor([tuple5])).cuda(), train_flag)
            r5 = float(y5.view(-1).cpu()[int(Sessen[i])])

            r=(r2+r3+r4+r5)/4.0

            for k in range(len(threshold)):
                if r >= threshold[k]:
                    if int(lables[i])==0:
                        FN[k] += 1
                    if int(lables[i]) == 1:
                        TN[k] += 1
                else:
                    if int(lables[i])==0:
                        TP[k] += 1
                    if int(lables[i]) == 1:
                        FP[k] += 1

    F1 = []
    TNR = []
    for k in range(len(TP)):
        precision[k] = float(TP[k]) / max((TP[k] + FP[k]), 1)
        recall[k] = float(TP[k]) / max((TP[k] + FN[k]), 1)
        F1.append(2 * precision[k] * recall[k] / max((precision[k] + recall[k]), 0.001))
        TNR.append(float(TN[k]) / max((TN[k] + FN[k]), 1))
    print(precision)
    print(recall)
    print(F1)
    print(TNR)
    f = open('Result/result_Bagging.csv', 'w', newline='')
    writer = csv.writer(f)
    writer.writerow(TP)
    writer.writerow(FP)
    writer.writerow(TN)
    writer.writerow(FN)
    writer.writerow(precision)
    writer.writerow(recall)
    writer.writerow(F1)
    writer.writerow(TNR)





if __name__=='__main__':
    window2 = 2
    window3 = 3
    window4 = 4
    window5 = 5
    max_window=5

    hidden_size = 128
    num_layers = 2
    num_keys = 394
    emb_dimension = 200
    topK = 10
    train_flag = False
    threshold = [0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 0.5, 0.45, 0.40, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1]
    torch.cuda.set_device(1)

    with torch.no_grad():
        model_window2 = G_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
        model_window3 = G_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
        model_window4 = G_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
        model_window5 = G_Model(emb_dimension, hidden_size, num_layers, num_keys, emb_dimension).cuda()
        model_window2.load_state_dict(torch.load('Checkpoint/G_model_mask_tmp_window2_Permutation.pkl'))
        model_window3.load_state_dict(torch.load('Checkpoint/G_model_mask_tmp.pkl'))
        model_window4.load_state_dict(torch.load('Checkpoint/G_model_mask_tmp_window4.pkl'))
        model_window5.load_state_dict(torch.load('Checkpoint/G_model_mask_tmp_window5.pkl'))
        print('模型初始化完成')
        TP = []
        FP = []
        FN = []
        TN = []
        precision = []
        recall = []
        for i in range(len(threshold)):
            TP.append(0)
            FP.append(0)
            FN.append(0)
            TN.append(0)
            precision.append(0)
            recall.append(0)

    Test()




